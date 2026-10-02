import numpy as np


def compute_channel_capacities(*, h: np.ndarray, sigma: float) -> np.ndarray:
    """
    Compute the channel capacities.

    :param h: (n_samples, n_antennas)-dimensional array with channels
    :param sigma: floating point number with std of noise
    :return: (n_samples)-dimensional array with channel capacities for channels h
    """
    signal = (
        np.abs(
            np.sum(
                h.conj() * (h / np.linalg.norm(h, axis=1, keepdims=True)),
                axis=1,
            )
        )
        ** 2
    )
    noise = sigma**2
    return np.log2(1 + signal / noise)


def compute_normalised_spectral_efficiencies(
    *,
    h: np.ndarray,
    combiners: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """
    Compute the normalised spectral efficiencies.

    :param h: (n_samples, n_antennas)-dimensional array with channels
    :param combiners: (n_samples, n_antennas, n_combiners)-dimensional array with combiners
    :param sigma: standard deviation of noise
    :return: (n_samples, n_combiners)-dimensional array with normalised spectral efficiencies for
        channels h when using given combiners. Normalization is w.r.t. channel capacity.
    """
    is_just_one_combiner_per_channel = len(combiners.shape) == 2
    if is_just_one_combiner_per_channel:
        combiners = combiners[..., np.newaxis]

    channel_capacities = compute_channel_capacities(h=h, sigma=sigma)[:, np.newaxis]
    signal = np.abs(np.sum(h.conj()[..., np.newaxis] * combiners, axis=1)) ** 2
    noise = sigma**2
    return np.log2(1 + signal / noise) / channel_capacities
