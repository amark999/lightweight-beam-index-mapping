import argparse
import copy
import os
import subprocess

import numpy as np
import pandas as pd
import torch

from generative_models.coupled_gmm import CoupledGMMy
from generative_models.gmm import GMM
from utils.dft_matrix import get_oversampled_dft_matrix_numpy
from utils.random_number_generation import crandn_torch
from utils.spectral_efficiency import compute_normalised_spectral_efficiencies

TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
rng = torch.Generator(device=TORCH_DEVICE)

MAX_EM_ALGORITHM_ITERATIONS = 1000

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        default="deepmimo_boston5g_3p5ghz",
        choices=["deepmimo_boston5g_3p5ghz", "quadriga_nlos"]
    )
    p.add_argument("--use-reduced-dataset", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--K", type=int, default=64)
    p.add_argument("--N", type=int, default=1, help="DFT codebook oversampling factor")
    p.add_argument("--snr-train-db", type=float, default=10)
    p.add_argument("--snr-test-db", type=float, default=10)
    p.add_argument("--sigma-r-test", type=float, default=0)
    return p.parse_args()

def main():
    args = parse_args()
    dataset = args.dataset
    use_reduced_dataset = args.use_reduced_dataset
    K = args.K
    N = args.N
    snr_train__dB = args.snr_train_db
    snr_test__dB = args.snr_test_db
    sigma_r_test = args.sigma_r_test

    dataset_selection = "reduced" if use_reduced_dataset else "complete"
    dataset_folder_path = os.sep.join(["datasets", dataset_selection, dataset])
    if not os.path.isdir(dataset_folder_path):
        print("Downloading complete dataset...")
        subprocess.run(
            ["bash", os.sep.join(["scripts", "download_dataset_complete.sh"])],
            check=True
        )

    train_data = np.load(os.sep.join([dataset_folder_path, "training-dataset.npz"]))
    test_data = np.load(os.sep.join([dataset_folder_path, "test-dataset.npz"]))

    H_train = train_data.get("h")
    r_train = train_data.get("r")
    H_test = test_data.get("h")
    r_test = test_data.get("r")

    L_train, N_tx, N_rx = H_train.shape
    L_test, _, _ = H_test.shape
    N_c = N * N_rx

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    out_folder = "results"
    os.makedirs(out_folder, exist_ok=True)

    F_tx = get_oversampled_dft_matrix_numpy(N_tx, oversampling_factor=1)
    F_rx = get_oversampled_dft_matrix_numpy(N_rx, oversampling_factor=N)

    r_train = torch.asarray(r_train, device=device, dtype=torch.float32)
    r_test = torch.asarray(r_test, device=device, dtype=torch.float32)

    # ==================================================================================
    # Apply optimal precoding at BS ----------------------------------------------------

    # Training data
    H_F_tx_train = H_train.swapaxes(-1, -2) @ F_tx[np.newaxis, :, :]
    i_star_train = np.argmax(np.linalg.norm(H_F_tx_train, axis=1), axis=1)
    f_tx_i_star_train = F_tx[:, i_star_train].T[:, :, np.newaxis]
    h_tilde_train = torch.asarray(
        (H_train.swapaxes(-1, -2) @ f_tx_i_star_train).squeeze(axis=2),
        device=device,
        dtype=torch.complex64,
    )
    h_tilde_train /= torch.sqrt(
        (torch.linalg.norm(h_tilde_train) ** 2) / (L_train * N_rx)
    )

    # Test data
    H_F_tx_test = H_test.swapaxes(-1, -2) @ F_tx[np.newaxis, :, :]
    i_star_test = np.argmax(np.linalg.norm(H_F_tx_test, axis=1), axis=1)
    f_tx_i_star_test = F_tx[:, i_star_test].T[:, :, np.newaxis]
    h_tilde_test = torch.asarray(
        (H_test.swapaxes(-1, -2) @ f_tx_i_star_test).squeeze(axis=2),
        device=device,
        dtype=torch.complex64,
    )
    h_tilde_test /= torch.sqrt((torch.linalg.norm(h_tilde_test) ** 2) / (L_test * N_rx))

    # ==================================================================================
    # Generate noisy data --------------------------------------------------------------

    # Training data
    sigma_n_train = 10 ** (-snr_train__dB / 20)
    y_tilde_train = (
        h_tilde_train
        + crandn_torch(size=h_tilde_train.shape, device=device, rng=rng) * sigma_n_train
    )

    # Test data
    sigma_n_test = 10 ** (-snr_test__dB / 20)
    y_s_tilde_exhaustive_search_test = []
    for _ in range(N_c):
        y_test = (
            h_tilde_test
            + crandn_torch(size=h_tilde_test.shape, device=device, rng=rng)
            * sigma_n_test
        )
        y_s_tilde_exhaustive_search_test.append(y_test)
    h_tilde_test = h_tilde_test.cpu().numpy()

    r_test += torch.randn(size=r_test.shape, device=device, generator=rng, dtype=torch.float32) * sigma_r_test / np.sqrt(2)

    # ==================================================================================
    # Train generative models ----------------------------------------------------------
    position_gmm = GMM(K=K, rng=rng, device=device)
    print("\nTraining Position GMM used for Clustering-based Fingerprinting")
    position_gmm.fit(x=r_train, max_iterations=MAX_EM_ALGORITHM_ITERATIONS)

    coupled_gmm = CoupledGMMy(
        K=K, training_snr__dB=snr_train__dB, rng=rng, device=device
    )
    print("\nTraining Coupled GMM")
    coupled_gmm.fit(
        y=y_tilde_train, r=r_train, max_iterations=MAX_EM_ALGORITHM_ITERATIONS
    )

    # ==================================================================================
    # Codebook assignment
    log_responsibilities, _ = position_gmm.e_step(x=r_train)
    k_l_clustering = torch.argmax(log_responsibilities, dim=1)

    v1s_clustering = torch.zeros(
        size=(N_rx, position_gmm.K), device=device, dtype=y_tilde_train.dtype
    )
    v1s_coupled_gmm = torch.zeros(
        size=(N_rx, coupled_gmm.K), device=device, dtype=y_tilde_train.dtype
    )

    for k in range(K):
        y = y_tilde_train[k_l_clustering == k, :]
        C_hat_y_tilde_k = (y.T @ y.conj()) / y.shape[0]
        try:
            evd_result = torch.linalg.eigh(C_hat_y_tilde_k)
            U = evd_result.eigenvectors
            v1s_clustering[:, k] = U[:, -1]
        except torch._C._LinAlgError:
            pass  # this happens because of no samples assigned to component k

        evd_result = torch.linalg.eigh(coupled_gmm.h_covariances[k, :, :])
        U = evd_result.eigenvectors
        v1s_coupled_gmm[:, k] = U[:, -1]

    j_k_clustering = np.argmax(
        np.abs(v1s_clustering.cpu().numpy().T.conj() @ F_rx), axis=1
    )
    j_k_coupled_gmm = np.argmax(
        np.abs(v1s_coupled_gmm.cpu().numpy().T.conj() @ F_rx), axis=1
    )

    coupled_gmm_with_refinement = copy.deepcopy(coupled_gmm)
    f_rx_j_k_coupled_gmm = F_rx[:, j_k_coupled_gmm].swapaxes(0, 1)
    C_h_tilde_k_T_plus_1 = torch.asarray(
        f_rx_j_k_coupled_gmm[:, :, np.newaxis] @ f_rx_j_k_coupled_gmm[:, np.newaxis, :].conj(),
        device=device, dtype=torch.complex64,
    )
    coupled_gmm_with_refinement.h_covariances = C_h_tilde_k_T_plus_1
    log_responsibilities, _ = coupled_gmm_with_refinement.e_step(y=y_tilde_train, sigma=sigma_n_train, r=r_train)
    coupled_gmm_with_refinement._m_step(y=y_tilde_train, r=r_train, log_gammas=log_responsibilities)
    coupled_gmm_with_refinement.h_covariances = C_h_tilde_k_T_plus_1

    # ==================================================================================
    # Performance Evaluation
    F_rx_indices_sorted_by_signal_power = np.argsort(
        np.abs(F_rx.T.conj() @ h_tilde_test.T), axis=0
    )[::-1, :].T
    j_genie_l = F_rx_indices_sorted_by_signal_power[:, 0]
    f_rx_j_genie_l = F_rx[:, j_genie_l].T

    signal_powers_exhaustive_search = np.zeros(shape=(L_test, N_c))
    for index, current_y_tilde_test in enumerate(y_s_tilde_exhaustive_search_test):
        signal_powers_exhaustive_search[:, index] = np.abs(
            F_rx[:, index].T.conj() @ current_y_tilde_test.cpu().numpy().T
        )
    j_exh_l = np.argmax(signal_powers_exhaustive_search, axis=1)
    f_rx_j_exh_l = F_rx[:, j_exh_l].T

    j_rnd_l = np.random.randint(0, N_c, L_test)
    f_rx_j_rnd_l = F_rx[:, j_rnd_l].T

    log_responsibilities, _ = position_gmm.e_step(x=r_test)
    k_l_clustering = torch.argmax(log_responsibilities, dim=1).cpu().numpy()
    j_k_l_clustering = j_k_clustering[k_l_clustering]
    f_rx_j_k_l_clustering = F_rx[:, j_k_l_clustering].T

    log_responsibilities, _ = coupled_gmm.e_step_r(r=r_test)
    k_l_coupled_gmm = torch.argmax(log_responsibilities, dim=1).cpu().numpy()
    j_k_l_coupled_gmm = j_k_coupled_gmm[k_l_coupled_gmm]
    f_rx_j_k_l_coupled_gmm = F_rx[:, j_k_l_coupled_gmm].T

    log_responsibilities, _ = coupled_gmm_with_refinement.e_step_r(r=r_test)
    k_l_coupled_gmm_with_refinement = torch.argmax(log_responsibilities, dim=1).cpu().numpy()
    j_k_l_coupled_gmm_with_refinement = j_k_coupled_gmm[k_l_coupled_gmm_with_refinement]
    f_rx_j_k_l_coupled_gmm_with_refinement = F_rx[:, j_k_l_coupled_gmm_with_refinement].T

    # ----------------------------------------------------------------------------------
    # Compute normalized mean spectral efficiencies
    mnse_genie = np.mean(
        compute_normalised_spectral_efficiencies(
            h=h_tilde_test, combiners=f_rx_j_genie_l, sigma=sigma_n_test
        ),
    )
    mnse_exhaustive_search = np.mean(
        compute_normalised_spectral_efficiencies(
            h=h_tilde_test, combiners=f_rx_j_exh_l, sigma=sigma_n_test
        ),
    )
    mnse_random_selection = np.mean(
        compute_normalised_spectral_efficiencies(
            h=h_tilde_test, combiners=f_rx_j_rnd_l, sigma=sigma_n_test
        ),
    )
    mnse_clustering = np.mean(
        compute_normalised_spectral_efficiencies(
            h=h_tilde_test, combiners=f_rx_j_k_l_clustering, sigma=sigma_n_test
        ),
    )
    mnse_coupled_gmm = np.mean(
        compute_normalised_spectral_efficiencies(
            h=h_tilde_test, combiners=f_rx_j_k_l_coupled_gmm, sigma=sigma_n_test
        ),
    )
    mnse_coupled_gmm_with_refinement = np.mean(
        compute_normalised_spectral_efficiencies(
            h=h_tilde_test,
            combiners=f_rx_j_k_l_coupled_gmm_with_refinement,
            sigma=sigma_n_test,
        ),
    )

    # ----------------------------------------------------------------------------------
    # Compute selection percentage values
    selection_percentages_exhaustive_search = np.sum(
        j_exh_l[:, np.newaxis] == F_rx_indices_sorted_by_signal_power, axis=0
    ) / L_test * 100
    selection_percentages_random_selection = np.sum(
        j_rnd_l[:, np.newaxis] == F_rx_indices_sorted_by_signal_power, axis=0
    ) / L_test * 100
    selection_percentages_clustering = np.sum(
        j_k_l_clustering[:, np.newaxis] == F_rx_indices_sorted_by_signal_power, axis=0
    ) / L_test * 100
    selection_percentages_coupled_gmm = np.sum(
        j_k_l_coupled_gmm[:, np.newaxis] == F_rx_indices_sorted_by_signal_power, axis=0
    ) / L_test * 100
    selection_percentages_coupled_gmm_with_refinement = np.sum(
        j_k_l_coupled_gmm_with_refinement[:, np.newaxis] == F_rx_indices_sorted_by_signal_power, axis=0
    ) / L_test * 100

    # ==================================================================================
    # Store values
    mnse_csv_file_rows = []
    selection_percentage_csv_file_rows = []
    # ----------------------------------------------------------------------------------
    # MnSE values
    mnse_csv_file_rows.append({"method": "genie", "train_snr_db": snr_train__dB, "test_snr_db": snr_test__dB, "mnse": mnse_genie, "sigma_r": sigma_r_test})
    mnse_csv_file_rows.append({"method": "exhaustive_search", "train_snr_db": snr_train__dB, "test_snr_db": snr_test__dB, "mnse": mnse_exhaustive_search, "sigma_r": sigma_r_test})
    mnse_csv_file_rows.append({"method": "random_selection", "train_snr_db": snr_train__dB, "test_snr_db": snr_test__dB, "mnse": mnse_random_selection, "sigma_r": sigma_r_test})
    mnse_csv_file_rows.append({"method": "clustering_based_fingerprinting", "train_snr_db": snr_train__dB, "test_snr_db": snr_test__dB, "mnse": mnse_clustering, "sigma_r": sigma_r_test})
    mnse_csv_file_rows.append({"method": "coupled_gmm", "train_snr_db": snr_train__dB, "test_snr_db": snr_test__dB, "mnse": mnse_coupled_gmm, "sigma_r": sigma_r_test})
    mnse_csv_file_rows.append({"method": "coupled_gmm_with_refinement", "train_snr_db": snr_train__dB, "test_snr_db": snr_test__dB, "mnse": mnse_coupled_gmm_with_refinement, "sigma_r": sigma_r_test})

    # ----------------------------------------------------------------------------------
    # Selection percentage values
    for spectral_efficiency_index in range(1, N_c + 1):
        selection_percentage_csv_file_rows.append({"method": "exhaustive_search", "train_snr_db": snr_train__dB, "test_snr_db": snr_test__dB, "spec_eff_rank": spectral_efficiency_index, "selection_percentage": selection_percentages_exhaustive_search[spectral_efficiency_index - 1], "sigma_r": sigma_r_test})
        selection_percentage_csv_file_rows.append({"method": "random_selection", "train_snr_db": snr_train__dB, "test_snr_db": snr_test__dB, "spec_eff_rank": spectral_efficiency_index, "selection_percentage": selection_percentages_random_selection[spectral_efficiency_index - 1], "sigma_r": sigma_r_test,})
        selection_percentage_csv_file_rows.append({"method": "clustering_based_fingerprinting", "train_snr_db": snr_train__dB, "test_snr_db": snr_test__dB, "spec_eff_rank": spectral_efficiency_index, "selection_percentage": selection_percentages_clustering[spectral_efficiency_index - 1], "sigma_r": sigma_r_test,})
        selection_percentage_csv_file_rows.append({"method": "coupled_gmm", "train_snr_db": snr_train__dB, "test_snr_db": snr_test__dB, "spec_eff_rank": spectral_efficiency_index, "selection_percentage": selection_percentages_coupled_gmm[spectral_efficiency_index - 1], "sigma_r": sigma_r_test,})
        selection_percentage_csv_file_rows.append({"method": "coupled_gmm_with_refinement", "train_snr_db": snr_train__dB, "test_snr_db": snr_test__dB, "spec_eff_rank": spectral_efficiency_index, "selection_percentage": selection_percentages_coupled_gmm_with_refinement[spectral_efficiency_index - 1], "sigma_r": sigma_r_test,})

    mnse_dataframe = pd.DataFrame(mnse_csv_file_rows)
    mnse_csv_file_path = os.path.join(out_folder, f"mean_nse_values.csv")
    mnse_dataframe.to_csv(mnse_csv_file_path, index=False)

    selection_percentage_dataframe = pd.DataFrame(selection_percentage_csv_file_rows)
    selection_percentage_dataframe["train_snr_db"] = selection_percentage_dataframe["train_snr_db"].astype("Int64")
    selection_percentage_csv_file_path = os.path.join(out_folder, f"selection_percentage_values.csv")
    selection_percentage_dataframe.to_csv(selection_percentage_csv_file_path, index=False)


if __name__ == "__main__":
    main()
