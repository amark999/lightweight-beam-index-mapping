import numpy as np


def get_oversampled_dft_matrix_numpy(size: int, oversampling_factor: int):
    """
    Returns a wide (size, size * oversampling_factor) matrix with DFT vector columns.
    """
    dft_columns = size * oversampling_factor

    D = np.zeros((size, dft_columns), dtype=complex)
    for m in range(size):
        for n in range(dft_columns):
            D[m, n] = (1 / np.sqrt(size)) * np.exp(
                -1j * 2 * np.pi * m * n / (size * oversampling_factor)
            )

    return D
