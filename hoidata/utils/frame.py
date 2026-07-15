"""Frame-level utilities for sequence alignment and masking.

Source: Text2HOI (https://github.com/JunukCha/Text2HOI.git)
  lib/utils/frame.py
"""

import numpy as np


def align_frame(total_dict):
    """Pad a dict of ragged sequence lists to the same length.

    Args:
        total_dict: dict mapping key → list of arrays, each array shaped
                    (nframes, ...).  All lists must have the same length
                    (number of sequences).
    Returns:
        final_dict: dict with same keys, values are np.ndarray of shape
                    (N, max_frames, ...) zero-padded to the longest sequence.
    """
    # Determine max sequence length from the first key
    first_value = next(iter(total_dict.values()))
    max_nframes = max(len(seq) for seq in first_value)
    ndata = len(first_value)

    final_dict = {}
    for key, value in total_dict.items():
        # Infer shape from first non-empty sequence
        sample = next(v for v in value if len(v) > 0)
        inner_shape = np.array(sample).shape[1:]
        final_data = np.zeros((ndata, max_nframes, *inner_shape), dtype=np.float32)
        for i, seq in enumerate(value):
            n = len(seq)
            if n == 0:
                continue
            final_data[i, :n] = seq
        final_dict[key] = final_data
    return final_dict


def get_valid_mask(is_lhand, is_rhand, nframes, valid_nframes):
    valid_mask_lhand = np.zeros(nframes)
    valid_mask_rhand = np.zeros(nframes)
    valid_mask_obj = np.zeros(nframes)
    valid_mask_lhand[:valid_nframes] = 1
    valid_mask_rhand[:valid_nframes] = 1
    valid_mask_obj[:valid_nframes] = 1
    if not is_lhand:
        valid_mask_lhand[:] = 0
    if not is_rhand:
        valid_mask_rhand[:] = 0
    return (
        valid_mask_lhand.astype(bool),
        valid_mask_rhand.astype(bool),
        valid_mask_obj.astype(bool),
    )
