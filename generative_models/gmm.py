import torch
from sklearn.cluster import KMeans
from torch import Generator

DEFAULT_CONVERGENCE_THRESHOLD = 1e-6
DEFAULT_COVARIANCE_REGULARIZATION = 1e-6


class GMM:
    """
    Gaussian Mixture Model.
    """

    def __init__(
        self,
        *,
        K: int,
        convergence_threshold: float = DEFAULT_CONVERGENCE_THRESHOLD,
        reg_covar: float = DEFAULT_COVARIANCE_REGULARIZATION,
        device: torch.device,
        rng: Generator,
    ) -> None:
        """
        Constructor of the Gaussian Mixture Model.

        :param K: number of Gaussian components
        :param convergence_threshold: stop training if mean log-likelihood changed by
            less than convergence_threshold.
        :param reg_covar: regularization parameter for covariance matrix
        :param device: torch device
        :param rng: random number generator
        """
        self.K = K

        self.convergence_threshold = convergence_threshold
        self.reg_covar = reg_covar

        self.device = device
        self.rng = rng

    @property
    def is_complex_valued(self) -> bool:
        return self.dtype.is_complex

    def _initialize_model_parameters(self, *, x: torch.Tensor) -> None:
        """
        Initialize the model parameters.

        :param x: (n_data_points, x_dimension)-dimensional tensor
        """
        self.x_dimension = x.shape[1]
        self.dtype = x.dtype

        self.weights = torch.zeros(size=(self.K,), device=self.device)
        self.means = torch.zeros(size=(self.K, self.x_dimension), device=self.device)
        self.covariances = torch.zeros(
            size=(self.K, self.x_dimension, self.x_dimension),
            device=self.device,
        )
        self.precision_cholesky = torch.zeros(
            size=(self.K, self.x_dimension, self.x_dimension),
            device=self.device,
        )
        if self.dtype.is_complex:
            self.means = self.means.to(dtype=self.dtype)
            self.covariances = self.covariances.to(dtype=self.dtype)
            self.precision_cholesky = self.precision_cholesky.to(dtype=self.dtype)

        # initialize responsibilities gammas by K-means clustering
        N = x.shape[0]
        log_gammas = torch.full(
            size=(N, self.K), fill_value=-float("inf"), device=self.device
        )
        x_stacked_real_imag = torch.cat(
            ([x.real, x.imag] if self.is_complex_valued else [x.real]), dim=1
        ).cpu()
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
            ).fit_predict(x_stacked_real_imag),
            device=self.device,
        )
        log_gammas[torch.arange(N, device=self.device), labels] = 0

        # update model parameters with M-step considering responsibilities gammas
        self._m_step(x=x, log_gammas=log_gammas)

    def fit(
        self,
        *,
        max_iterations: int,
        x: torch.Tensor,
    ) -> None:
        """
        Training of the Gaussian Mixture Model.

        :param max_iterations: maximum number of iterations for EM algorithm
        :param x: (n_data_points, x_dimension)-dimensional tensor
        """
        self._initialize_model_parameters(x=x)

        previous_log_likelihood = -torch.inf

        for iteration in range(max_iterations):
            log_gammas, log_likelihood = self.e_step(x=x)

            log_likelihood_increase = torch.real(
                log_likelihood - previous_log_likelihood
            )
            print(
                f"\riteration: {iteration + 1}, "
                f"log-likelihood increase: {log_likelihood_increase:.7f}",
                end="",
            )

            self._m_step(x=x, log_gammas=log_gammas)

            if (
                torch.abs(log_likelihood - previous_log_likelihood)
                < self.convergence_threshold
            ):
                print(f"\nConverged after {iteration + 1} iterations")
                break

            previous_log_likelihood = log_likelihood

            if iteration == max_iterations - 1:
                print("\nMaximum number of iterations reached.")

    def e_step(self, *, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the log of the responsibilities and the mean log-likelihood value.

        :param x: (n_data_points, x_dimension)-dimensional tensor
        :return: tuple containing two values:
            - (n_data_points, K)-dimensional tensor, containing the log of
                responsibilities log_gammas for each component of each data point
            - scalar mean log-likelihood value per data point
        """
        N = x.shape[0]

        log_pdf_values = torch.zeros(size=(N, self.K), device=self.device)
        for k in range(self.K):
            log_pdf_values[:, k] = self._compute_log_gaussian_pdf_value(
                data=x,
                mean=self.means[k, :],
                precision_cholesky=self.precision_cholesky[k, :, :],
            )

        log_pi_k = torch.log(self.weights)

        log_sum_weighted_pdf_values = torch.logsumexp(log_pi_k + log_pdf_values, dim=1)

        log_gammas = (
            log_pi_k[torch.newaxis, :]
            + log_pdf_values
            - log_sum_weighted_pdf_values[:, torch.newaxis]
        )

        return log_gammas, torch.mean(log_sum_weighted_pdf_values)

    def _m_step(
        self,
        *,
        x: torch.Tensor,
        log_gammas: torch.Tensor,
    ) -> None:
        """
        Updates the weights, means, covariances, and precision Cholesky factors.

        :param x: (n_data_points, x_dimension)-dimensional tensor
        :param log_gammas: (n_data_points, K)-dimensional log-responsibilities
        """
        gammas = torch.exp(log_gammas)
        N_k = torch.sum(gammas, dim=0) + 10 * torch.finfo(gammas.dtype).eps
        N_k_inv = 1 / N_k
        N = x.shape[0]

        self.means = (
            N_k_inv[:, torch.newaxis] * torch.matmul(gammas.T.to(self.dtype), x)
        ).to(self.dtype)
        for k in range(self.K):
            x_minus_mean = x - self.means[k, :]

            self.covariances[k, :, :] = (
                N_k_inv[k] * (gammas[:, k] * x_minus_mean.T) @ x_minus_mean.conj()
            )
            self.covariances[k, :, :] += (
                torch.eye(self.x_dimension, device=self.device) * self.reg_covar
            )

        self._update_precision_cholesky()

        self.weights = N_k / N

    def _compute_log_gaussian_pdf_value(
        self,
        *,
        data: torch.Tensor,
        mean: torch.Tensor,
        precision_cholesky: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the log of the Gaussian pdf value at data.

        :param data: (n_data_points, data_dimension)-dimensional tensor
        :param mean: (data_dimension)-dimensional mean
        :param precision_cholesky: (data_dimension, data_dimension)-dimensional
            Cholesky decomposition of precision matrix
        :return: (n_data_points)-dimensional tensor of log Gaussian probability values
        """
        data_dimension = data.shape[1]

        # Compute first term
        log_mul_term = -data_dimension * torch.log(
            torch.tensor(torch.pi * (1 if self.is_complex_valued else 2))
        )

        # Compute second term
        log_det_term = 2 * torch.real(
            torch.sum(torch.log(precision_cholesky.diagonal()))
        )

        # Compute third term: (data - mu)^T covariance^(-1) (data - mu)
        data_minus_mean = data - mean
        # use z = (data - mu) precision_cholesky^H, and compute ||z||^2
        z = torch.matmul(data_minus_mean, precision_cholesky.conj())
        log_exp_term = -torch.linalg.norm(z, dim=1) ** 2

        if self.is_complex_valued:
            return log_mul_term + log_det_term + log_exp_term
        else:
            return 0.5 * (log_mul_term + log_det_term + log_exp_term)

    def _update_precision_cholesky(self) -> None:
        """
        Updates the stored Cholesky decomposition of the precision matrix.
        """
        for k in range(self.K):
            covariance_cholesky = torch.linalg.cholesky(self.covariances[k, :, :])
            self.precision_cholesky[k, :, :] = torch.linalg.solve_triangular(
                covariance_cholesky,
                torch.eye(self.x_dimension, device=self.device),
                upper=False,
            ).T.conj()
