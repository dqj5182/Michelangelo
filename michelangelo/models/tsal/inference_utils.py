# -*- coding: utf-8 -*-

import torch
from tqdm import tqdm
from einops import repeat
import numpy as np
from typing import Callable, Tuple, List, Union, Optional
from skimage import measure

from external.Michelangelo.michelangelo.graphics.primitives import generate_dense_grid_points


@torch.no_grad()
def extract_geometry(geometric_func: Callable,
                     device: torch.device,
                     batch_size: int = 1,
                     bounds: Union[Tuple[float], List[float], float] = (-1.25, -1.25, -1.25, 1.25, 1.25, 1.25),
                     octree_depth: int = 7,
                     num_chunks: int = 10000,
                     disable: bool = True):
    """

    Args:
        geometric_func:
        device:
        bounds:
        octree_depth:
        batch_size:
        num_chunks:
        disable:

    Returns:

    """

    if isinstance(bounds, float):
        bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]

    bbox_min = np.array(bounds[0:3])
    bbox_max = np.array(bounds[3:6])
    bbox_size = bbox_max - bbox_min

    xyz_samples, grid_size, length = generate_dense_grid_points(
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        octree_depth=octree_depth,
        indexing="ij"
    )
    xyz_samples = torch.FloatTensor(xyz_samples)

    batch_logits = []

    for start in range(0, xyz_samples.shape[0], num_chunks):
        queries = xyz_samples[start: start + num_chunks, :].to(device)
        batch_queries = repeat(queries, "p c -> b p c", b=batch_size)

        logits = geometric_func(batch_queries)
        batch_logits.append(logits.cpu())

    grid_logits = torch.cat(batch_logits, dim=1).view((batch_size, grid_size[0], grid_size[1], grid_size[2])).numpy()

    mesh_v_f = []
    has_surface = np.zeros((batch_size,), dtype=np.bool_)

    return grid_logits, bbox_size, bbox_min, grid_size
