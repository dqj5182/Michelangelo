# -*- coding: utf-8 -*-

from typing import List, Tuple, Dict, Optional
from omegaconf import DictConfig

import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
import pytorch_lightning as pl
from typing import Union
from functools import partial

from external.Michelangelo.michelangelo.utils import instantiate_from_config

from .inference_utils import extract_geometry
from .tsal_base import (
    AlignedShapeAsLatentModule,
    ShapeAsLatentModule,
    Latent2MeshOutput,
    AlignedMeshOutput
)


class AlignedShapeAsLatentPLModule(pl.LightningModule):

    def __init__(self, *,
                 shape_module_cfg,
                 aligned_module_cfg,
                 loss_cfg,
                 optimizer_cfg: Optional[DictConfig] = None,
                 ckpt_path: Optional[str] = None,
                 ignore_keys: Union[Tuple[str], List[str]] = ()):

        super().__init__()

        shape_model: ShapeAsLatentModule = instantiate_from_config(
            shape_module_cfg, device=None, dtype=None
        )
        self.model: AlignedShapeAsLatentModule = instantiate_from_config(
            aligned_module_cfg, shape_model=shape_model
        )

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.save_hyperparameters()

    def set_shape_model_only(self):
        self.model.set_shape_model_only()

    @property
    def latent_shape(self):
        return self.model.shape_model.latent_shape

    @property
    def zero_rank(self):
        if self._trainer:
            zero_rank = self.trainer.local_rank == 0
        else:
            zero_rank = True

        return zero_rank

    def init_from_ckpt(self, path, ignore_keys=()):
        state_dict = torch.load(path, map_location="cpu")["state_dict"]

        keys = list(state_dict.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del state_dict[k]

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def forward(self,
                surface: torch.FloatTensor,
                image: torch.FloatTensor,
                text: torch.FloatTensor,
                volume_queries: torch.FloatTensor):

        """

        Args:
            surface (torch.FloatTensor):
            image (torch.FloatTensor):
            text (torch.FloatTensor):
            volume_queries (torch.FloatTensor):

        Returns:

        """

        embed_outputs, shape_z = self.model(surface, image, text)

        shape_zq, posterior = self.model.shape_model.encode_kl_embed(shape_z)
        latents = self.model.shape_model.decode(shape_zq)
        logits = self.model.shape_model.query_geometry(volume_queries, latents)

        return embed_outputs, logits, posterior

    def encode(self, surface: torch.FloatTensor, sample_posterior=True):

        pc = surface[..., 0:3]
        feats = surface[..., 3:6]

        shape_embed, shape_zq, posterior = self.model.shape_model.encode(
            pc=pc, feats=feats, sample_posterior=sample_posterior
        )

        return shape_zq

    def decode(self,
               z_q,
               bounds: Union[Tuple[float], List[float], float] = 1.1,
               octree_depth: int = 7,
               num_chunks: int = 10000) -> List[Latent2MeshOutput]:

        latents = self.model.shape_model.decode(z_q)  # latents: [bs, num_latents, dim]
        outputs = self.latent2mesh(latents, bounds=bounds, octree_depth=octree_depth, num_chunks=num_chunks)

        return outputs

    def latent2mesh(self,
                    latents: torch.FloatTensor,
                    bounds: Union[Tuple[float], List[float], float] = 1.1,
                    octree_depth: int = 7,
                    num_chunks: int = 10000) -> List[Latent2MeshOutput]:

        """

        Args:
            latents: [bs, num_latents, dim]
            bounds:
            octree_depth:
            num_chunks:

        Returns:
            mesh_outputs (List[MeshOutput]): the mesh outputs list.

        """

        outputs = {'grid_logits': [], 'bbox_sizes': [], 'bbox_mins': [], 'grid_sizes': []}

        geometric_func = partial(self.model.shape_model.query_geometry, latents=latents)

        # 2. decode geometry
        device = latents.device
        # mesh_v_f, has_surface, grid_logits = extract_geometry(
        #     geometric_func=geometric_func,
        #     device=device,
        #     batch_size=len(latents),
        #     bounds=bounds,
        #     octree_depth=octree_depth,
        #     num_chunks=num_chunks,
        #     disable=not True #self.zero_rank
        # )
        grid_logits, bbox_sizes, bbox_mins, grid_sizes = extract_geometry(
            geometric_func=geometric_func,
            device=device,
            batch_size=len(latents),
            bounds=bounds,
            octree_depth=octree_depth,
            num_chunks=num_chunks,
            disable=not True #self.zero_rank
        )

        # 3. decode texture
        # for i, ((mesh_v, mesh_f), is_surface, grid_logits) in enumerate(zip(mesh_v_f, has_surface, grid_logits)):
        for i, (grid_logit, bbox_size, bbox_min, grid_size) in enumerate(zip(grid_logits, bbox_sizes, bbox_mins, grid_sizes)):
            # if not is_surface:
            #     outputs.append(None)
            #     continue

            # out = Latent2MeshOutput()
            # out.mesh_v = mesh_v
            # out.mesh_f = mesh_f
            # out.occp = grid_logit


            outputs['grid_logits'].append(grid_logit)
            outputs['bbox_sizes'].append(bbox_size)
            outputs['bbox_mins'].append(bbox_min)
            outputs['grid_sizes'].append(grid_size)
        return outputs

