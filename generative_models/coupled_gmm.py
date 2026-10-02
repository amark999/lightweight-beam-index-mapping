from typing import Optional

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import Generator

DEFAULT_CONVERGENCE_THRESHOLD = 1e-6
DEFAULT_COVARIANCE_REGULARIZATION = 1e-6


class CoupledGMMy:
    """
    Coupled GMM trained on pilot observations y = Ax + n.

    Implementation of the Coupled GMM from [1], with extensions.
    [1] F. Weißer, A. Kasibovic, B. Bock, and W. Utschick, «No Pilots, No Problem: A
    Generative Model for Position-based Downlink Precoder Design».
    """

    def __init__(
        self,
        *,
        K: int,
        training_snr__dB: float,
        convergence_threshold: float = DEFAULT_CONVERGENCE_THRESHOLD,
        reg_covar: float = DEFAULT_COVARIANCE_REGULARIZATION,
        device: torch.device,
        rng: Generator,
    ) -> None:
        """
        Constructor of the Coupled Gaussian Mixture Model.

        :param K: number of Gaussian components
        :param training_snr__dB: signal-to-noise ratio in dB of training data
        :param convergence_threshold: stop training if mean log-likelihood changed by
            less than convergence_threshold.
        :param reg_covar: regularization parameter for covariance matrix
        :param device: torch device
        :param rng: random number generator
        """
        self.K = K
        self.training_snr__dB = training_snr__dB

        self.convergence_threshold = convergence_threshold
        self.reg_covar = reg_covar

        self.device = device
        self.rng = rng

    def _initialize_model_parameters(self, *, y: torch.Tensor, r: torch.Tensor) -> None:
        """
        Initialize the model parameters.

        :param y: (n_data_points, y_dimension)-dimensional tensor
        :param r: (n_data_points, r_dimension)-dimensional tensor
        """
        self.y_dtype = y.dtype
        self.y_dimension = y.shape[1]
        self.h_dimension = self.y_dimension
        self.r_dtype = r.dtype
        self.r_dimension = r.shape[1]

        self.training_sigma = 10 ** (-self.training_snr__dB / 20)
        self.training_Sigma_n = (
            torch.eye(self.y_dimension, device=self.device, dtype=self.y_dtype)
            * self.training_sigma**2
        )

        self.weights = torch.zeros(size=(self.K,), device=self.device)
        self.h_covariances = torch.zeros(
            size=(self.K, self.h_dimension, self.h_dimension),
            device=self.device,
            dtype=self.y_dtype,
        )
        self.y_training_covariances = torch.zeros(
            size=(self.K, self.y_dimension, self.y_dimension),
            device=self.device,
            dtype=self.y_dtype,
        )
        self.r_means = torch.zeros(
            size=(self.K, self.r_dimension), device=self.device, dtype=self.r_dtype
        )
        self.r_covariances = torch.zeros(
            size=(self.K, self.r_dimension, self.r_dimension),
            device=self.device,
            dtype=self.r_dtype,
        )
        self.r_precision_cholesky = torch.zeros(
            size=(self.K, self.r_dimension, self.r_dimension),
            device=self.device,
            dtype=self.r_dtype,
        )

        # initialize responsibilities gammas by K-means clustering
        N = y.shape[0]
        log_gammas = torch.full(
            size=(N, self.K), fill_value=-float("inf"), device=self.device
        )
        y_stacked_real_imag = torch.cat([y.real, y.imag], dim=1).cpu()
        labels = torch.tensor(
            KMeans(
                n_clusters=self.K,
                random_state=int(
                    torch.randint(
                        low=0,
                        high=2**32 - 1,
                        generator=self.rng,
                        size=(1, 1),
                        device=self.device,
                    )
                ),
            ).fit_predict(y_stacked_real_imag),
            device=self.device,
        )
        log_gammas[torch.arange(N, device=self.device), labels] = 0

        # update model parameters with M-step considering responsibilities gammas
        self._m_step(y=y, r=r, log_gammas=log_gammas)

    def fit(
        self,
        *,
        max_iterations: int,
        y: torch.Tensor,
        r: torch.Tensor,
    ) -> None:
        """
        Training of the Coupled Gaussian Mixture Model.

        :param max_iterations: maximum number of iterations for EM algorithm
        :param y: (n_data_points, y_dimension)-dimensional tensor
        :param r: (n_data_points, r_dimension)-dimensional tensor
        """
        self._initialize_model_parameters(y=y, r=r)

        previous_log_likelihood = -torch.inf

        for iteration in range(max_iterations):
            for k in range(self.K):
                self.y_training_covariances[k, :, :] = (
                    self.h_covariances[k, :, :] + self.training_Sigma_n
                )

            log_gammas, log_likelihood = self.e_step(
                y=y, sigma=self.training_sigma, r=r
            )

            log_likelihood_increase = torch.real(
                log_likelihood - previous_log_likelihood
            )
            print(
                f"\riteration: {iteration + 1}, "
                f"log-likelihood increase: {log_likelihood_increase:.7f}",
                end="",
            )

            self._m_step(y=y, r=r, log_gammas=log_gammas)

            if (
                torch.abs(log_likelihood - previous_log_likelihood)
                < self.convergence_threshold
            ):
                print(f"\nConverged after {iteration + 1} iterations")
                break

            previous_log_likelihood = log_likelihood

            if iteration == max_iterations - 1:
                print("\nMaximum number of iterations reached.")

    def e_step(
        self,
        *,
        y: torch.Tensor,
        sigma: float,
        r: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the log of the responsibilities and the mean log-likelihood value.

        :param y: (n_data_points, y_dimension)-dimensional tensor
        :param sigma: noise standard deviation of the observations y
        :param r: (n_data_points, r_dimension)-dimensional tensor
        :return: tuple containing two values:
            - (n_data_points, K)-dimensional tensor, containing the log of
                responsibilities log_gammas for each component of each data point
            - scalar mean log-likelihood value per data point
        """
        N = y.shape[0]

        log_y_pdf_values = torch.zeros(size=(N, self.K), device=self.device)
        log_r_pdf_values = torch.zeros(size=(N, self.K), device=self.device)
        for k in range(self.K):
            y_covariance_cholesky = self._cholesky(
                self.h_covariances[k, :, :]
                + sigma**2 * torch.eye(self.y_dimension, device=self.device)
            )
            y_precision_cholesky = torch.linalg.solve_triangular(
                y_covariance_cholesky,
                torch.eye(self.y_dimension, device=self.device),
                upper=False,
            ).T.conj()
            log_y_pdf_values[:, k] = self._compute_log_gaussian_pdf_value(
                data=y,
                precision_cholesky=y_precision_cholesky,
            )
            log_r_pdf_values[:, k] = self._compute_log_gaussian_pdf_value(
                data=r,
                mean=self.r_means[k, :],
                precision_cholesky=self.r_precision_cholesky[k, :, :],
            )

        log_pi_k = torch.log(self.weights)

        log_sum_weighted_pdf_values = torch.logsumexp(
            log_pi_k + log_y_pdf_values + log_r_pdf_values, dim=1
        )

        log_gammas = (
            log_pi_k[torch.newaxis, :]
            + log_y_pdf_values
            + log_r_pdf_values
            - log_sum_weighted_pdf_values[:, torch.newaxis]
        )

        return log_gammas, torch.mean(log_sum_weighted_pdf_values)

    def e_step_r(self, *, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the log of the responsibilities and the mean log-likelihood value,
        computed from the position information r only.

        :param r: (n_data_points, r_dimension)-dimensional tensor
        :return: tuple containing two values:
            - (n_data_points, K)-dimensional tensor, containing the log of
                responsibilities log_gammas for each component of each data point
            - scalar mean log-likelihood value per data point
        """
        N = r.shape[0]

        log_r_pdf_values = torch.zeros(size=(N, self.K), device=self.device)
        for k in range(self.K):
            log_r_pdf_values[:, k] = self._compute_log_gaussian_pdf_value(
                data=r,
                mean=self.r_means[k, :],
                precision_cholesky=self.r_precision_cholesky[k, :, :],
            )

        log_pi_k = torch.log(self.weights)

        log_sum_weighted_pdf_values = torch.logsumexp(
            log_pi_k + log_r_pdf_values, dim=1
        )

        log_gammas = (
            log_pi_k[torch.newaxis, :]
            + log_r_pdf_values
            - log_sum_weighted_pdf_values[:, torch.newaxis]
        )

        return log_gammas, torch.mean(log_sum_weighted_pdf_values)

    def _m_step(
        self,
        *,
        y: torch.Tensor,
        r: torch.Tensor,
        log_gammas: torch.Tensor,
    ) -> None:
        """
        Updates the weights, means, covariances, and precision Cholesky factors.

        :param y: (n_data_points, y_dimension)-dimensional tensor
        :param r: (n_data_points, r_dimension)-dimensional tensor
        :param log_gammas: (n_data_points, K)-dimensional log-responsibilities
        """
        gammas = torch.exp(log_gammas)
        N_k = torch.sum(gammas, dim=0) + 10 * torch.finfo(gammas.dtype).eps
        N_k_inv = 1 / N_k
        N = y.shape[0]

        self.r_means = N_k_inv[:, torch.newaxis] * torch.matmul(
            gammas.T.to(dtype=self.r_dtype), r
        )
        for k in range(self.K):
            r_minus_mean = r - self.r_means[k, :]

            self.y_training_covariances[k, :, :] = (
                N_k_inv[k] * (gammas[:, k] * y.T) @ y.conj()
            )
            self.y_training_covariances[k, :, :] += (
                torch.eye(self.y_dimension, device=self.device) * self.reg_covar
            )
            self.r_covariances[k, :, :] = (
                N_k_inv[k] * (gammas[:, k] * r_minus_mean.T) @ r_minus_mean.conj()
            )
            self.r_covariances[k, :, :] += (
                torch.eye(self.r_dimension, device=self.device) * self.reg_covar
            )

            self.h_covariances[k, :, :] = self._project_on_positive_definite_matrix(
                self.y_training_covariances[k, :, :] - self.training_Sigma_n
            )

        self._update_precision_cholesky()

        self.weights = N_k / N

    def _compute_log_gaussian_pdf_value(
        self,
        *,
        data: torch.Tensor,
        mean: Optional[torch.Tensor] = None,
        precision_cholesky: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the log of the Gaussian pdf value at data.

        :param data: (n_data_points, data_dimension)-dimensional tensor
        :param mean: (data_dimension)-dimensional mean of the Gaussian distribution, if
            not specified it is assumed to be zero
        :param precision_cholesky: (data_dimension, data_dimension)-dimensional
            Cholesky decomposition of precision matrix of the Gaussian distribution
        :return: (n_data_points)-dimensional tensor of log Gaussian probability values
        """
        data_dimension = data.shape[1]

        # Compute first term
        log_mul_term = -data_dimension * torch.log(
            torch.tensor(torch.pi * (1 if data.dtype.is_complex else 2))
        )

        # Compute second term
        log_det_term = 2 * torch.real(
            torch.sum(torch.log(precision_cholesky.diagonal()))
        )

        # Compute third term: (data - mu)^T covariance^(-1) (data - mu)
        data_minus_mean = data - mean if mean is not None else data
        # use z = (data - mu) precision_cholesky^H, and compute ||z||^2
        z = torch.matmul(data_minus_mean, precision_cholesky.conj())
        log_exp_term = -torch.linalg.norm(z, dim=1) ** 2

        if data.dtype.is_complex:
            return log_mul_term + log_det_term + log_exp_term
        else:
            return 0.5 * (log_mul_term + log_det_term + log_exp_term)

    def _update_precision_cholesky(self) -> None:
        """
        Updates the stored Cholesky decomposition of the precision matrix of r.
        """
        for k in range(self.K):
            r_covariance_cholesky = self._cholesky(self.r_covariances[k, :, :])
            self.r_precision_cholesky[k, :, :] = torch.linalg.solve_triangular(
                r_covariance_cholesky,
                torch.eye(self.r_dimension, device=self.device),
                upper=False,
            ).T.conj()

    def _project_on_positive_definite_matrix(self, M: torch.Tensor) -> torch.Tensor:
        """
        Projects a Hermitian matrix onto the positive definite matrices by clipping
        its eigenvalues from below at reg_covar.

        :param M: (dimension, dimension)-dimensional tensor
        :return: (dimension, dimension)-dimensional positive definite tensor
        """
        try:
            L, Q = torch.linalg.eigh(M)
        except torch._C._LinAlgError:
            L, Q = np.linalg.eig(M.cpu().numpy())
            L = torch.asarray(L, device=self.device)
            Q = torch.asarray(Q, device=self.device)
        L[L.real < self.reg_covar] = self.reg_covar
        return Q @ torch.diag(L).to(dtype=Q.dtype) @ Q.T.conj()

    def _cholesky(self, M: torch.Tensor) -> torch.Tensor:
        """
        Computes the Cholesky decomposition of M. If M is not positive definite, a
        diagonal regularization (starting at reg_covar and doubling after every failed
        attempt) is added until the decomposition succeeds.

        :param M: (dimension, dimension)-dimensional tensor
        :return: (dimension, dimension)-dimensional lower triangular tensor
        """
        try:
            return torch.linalg.cholesky(M)
        except torch._C._LinAlgError:
            regularization = self.reg_covar
            M = M.cpu().numpy()
            while True:
                try:
                    L = np.linalg.cholesky(M)
                    break
                except np.linalg.LinAlgError:
                    M = M + np.eye(M.shape[0]) * regularization
                    regularization = regularization * 2
            return torch.asarray(L, device=self.device)
