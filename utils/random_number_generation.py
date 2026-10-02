from math import sqrt
from typing import Sequence, Union

import torch


def crandn_torch(
    size: Sequence[Union[int]],
    device: torch.device,
    rng: torch.Generator = None,
    dtype: torch.dtype = None,
):
    if rng is None:
        rng = torch.Generator(device=device)
    return (
        torch.randn(size=size, device=device, dtype=dtype, generator=rng)
        + 1j * torch.randn(size=size, device=device, dtype=dtype, generator=rng)
    ) / sqrt(2)
