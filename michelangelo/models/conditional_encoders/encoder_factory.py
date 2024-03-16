# -*- coding: utf-8 -*-
import os

import torch
import torch.nn as nn
from torchvision import transforms
from transformers import CLIPModel, CLIPTokenizer
from collections import OrderedDict


class AbstractEncoder(nn.Module):
    embedding_dim: int

    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class FrozenCLIPImageGridEmbedder(AbstractEncoder):

    def __init__(
            self,
            version="openai/clip-vit-large-patch14",
            device="cuda",
            zero_embedding_radio=0.1,
    ):
        super().__init__()

        self.device = device

        self.clip_dict = OrderedDict()
        self.clip_name = os.path.split(version)[-1]

        clip_model: CLIPModel = CLIPModel.from_pretrained(version)
        clip_model.text_model = None
        clip_model.text_projection = None
        clip_model = clip_model.eval()
        for param in self.parameters():
            param.requires_grad = False
        self.clip_dict[self.clip_name] = clip_model.to(self.device)

        self.transform = transforms.Compose(
            [
                transforms.Resize(224, transforms.InterpolationMode.BILINEAR, antialias=True),
                transforms.CenterCrop(224),  # crop a (224, 224) square
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        self.zero_embedding_radio = zero_embedding_radio
        self.embedding_dim = clip_model.vision_embed_dim

        self._move_flag = True

    @property
    def clip(self):
        return self.clip_dict[self.clip_name]

    def move(self):
        import pdb; pdb.set_trace()
        if self._move_flag:
            return

        self.clip_dict[self.clip_name] = self.clip_dict[self.clip_name] #.to(self.device)
        self._move_flag = True

    def unconditional_embedding(self, batch_size):
        zero = torch.zeros(
            batch_size,
            self.clip.vision_model.embeddings.num_positions,
            self.embedding_dim,
            device=self.device,
            dtype=self.clip.visual_projection.weight.dtype,
        )
        return zero

    def forward(self, image, value_range=(-1, 1), zero_embedding_radio=0):
        import time

        start = time.time()
        if value_range is not None:
            low, high = value_range
            image = (image - low) / (high - low)

        image = image.to(self.clip.device, dtype=self.clip.visual_projection.weight.dtype)
        end = time.time()
        print('Image processing:', str(end-start))

        start = time.time()
        self.clip = self.clip
        try:
            z = self.clip.vision_model(self.transform(image).to(self.clip.device)).last_hidden_state
        except:
            import pdb; pdb.set_trace()
        end = time.time()
        print('CLIP vision model:', str(end-start))

        if zero_embedding_radio > 0:
            mask = torch.rand((len(image), 1, 1), device=z.device, dtype=z.dtype) >= zero_embedding_radio
            z = z * mask.to(z)

        return z

    def encode(self, image):
        return self(image, zero_embedding_radio=self.zero_embedding_radio)