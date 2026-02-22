"""
gaussfun.py  —  mirrors gaussfun.m

Returns Gauss point locations and weights for 2D quadrilateral elements.
"""

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float64


def gaussfun(ngp: int):
    """
    Return Gauss point locations and weights.

    Parameters
    ----------
    ngp : 4  (2×2 rule)  or  9  (3×3 rule)

    Returns
    -------
    gp_locs : [ngp, 2]  (s, t) coordinates in [-1, 1]^2
    gp_wts  : [ngp]     weights
    """
    if ngp == 4:
        pt      = 0.577350269189626          # 1/sqrt(3)
        locs    = [[-pt, -pt], [pt, -pt], [pt, pt], [-pt, pt]]
        weights = [1.0, 1.0, 1.0, 1.0]

    elif ngp == 9:
        wt1 = 5.0 / 9.0
        wt2 = 8.0 / 9.0
        pt1 = (3.0 / 5.0) ** 0.5
        pt2 = 0.0
        locs = [
            [-pt1, -pt1], [-pt1,  pt2], [-pt1,  pt1],
            [ pt2, -pt1], [ pt2,  pt2], [ pt2,  pt1],
            [ pt1, -pt1], [ pt1,  pt2], [ pt1,  pt1],
        ]
        weights = [
            wt1*wt1, wt1*wt2, wt1*wt1,
            wt2*wt1, wt2*wt2, wt2*wt1,
            wt1*wt1, wt1*wt2, wt1*wt1,
        ]

    else:
        raise ValueError(f"gaussfun: unsupported ngp={ngp}. Use 4 or 9.")

    gp_locs = torch.tensor(locs,    dtype=DTYPE, device=DEVICE)
    gp_wts  = torch.tensor(weights, dtype=DTYPE, device=DEVICE)
    return gp_locs, gp_wts
