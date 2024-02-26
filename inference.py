# -*- coding: utf-8 -*-
import os
import argparse

from einops import repeat, rearrange
import trimesh
import cv2

import torch
import pytorch_lightning as pl

from external.Michelangelo.michelangelo.utils.misc import get_config_from_file, instantiate_from_config


def load_model():
    model_config = get_config_from_file('../external/Michelangelo/configs/image_cond_diffuser_asl/image-ASLDM-256.yaml')
    if hasattr(model_config, "model"):
        model_config = model_config.model 
    model = instantiate_from_config(model_config, ckpt_path='../external/Michelangelo/checkpoints/image_cond_diffuser_asl/image-ASLDM-256.ckpt')
    return model

def prepare_image(image_pt, number_samples=1):
    image_pt = image_pt / 255 * 2 - 1
    image_pt = rearrange(image_pt, "b h w c -> b c h w")
    return image_pt

def save_output(args, mesh_outputs):
    os.makedirs(args.output_dir, exist_ok=True)
    for i, mesh in enumerate(mesh_outputs):
        mesh.mesh_f = mesh.mesh_f[:, ::-1]
        mesh_output = trimesh.Trimesh(mesh.mesh_v, mesh.mesh_f)

        name = str(i) + "_out_mesh.obj"
        mesh_output.export(os.path.join(args.output_dir, name), include_normals=True)

    print(f'-----------------------------------------------------------------------------')
    print(f'>>> Finished and mesh saved in {args.output_dir}')
    print(f'-----------------------------------------------------------------------------')        

    return 0

def image2mesh(args, model, guidance_scale=7.5, box_v=1.1, octree_depth=7):

    sample_inputs = {
        "image": prepare_image(image)
    }
    
    mesh_outputs = model.sample(
        sample_inputs,
        sample_times=1,
        guidance_scale=guidance_scale,
        return_intermediates=False,
        bounds=[-box_v, -box_v, -box_v, box_v, box_v, box_v],
        octree_depth=octree_depth,
    )[0]
    
    save_output(args, mesh_outputs)
    
    return 0

task_dick = {
    'image2mesh': image2mesh,
}

if __name__ == "__main__":
    '''
    1. Reconstruct point cloud
    2. Image-conditioned generation
    3. Text-conditioned generation
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default='image2mesh')
    parser.add_argument("--config_path", type=str, default='configs/image_cond_diffuser_asl/image-ASLDM-256.yaml')
    parser.add_argument("--ckpt_path", type=str, default='checkpoints/image_cond_diffuser_asl/image-ASLDM-256.ckpt')
    parser.add_argument("--image_path", type=str, default='example_data/image/car.jpg')
    parser.add_argument("--output_dir", type=str, default='./output')
    parser.add_argument("-s", "--seed", type=int, default=0)
    args = parser.parse_args()
    
    pl.seed_everything(args.seed)

    print(f'-----------------------------------------------------------------------------')
    print(f'>>> Running {args.task}')
    args.output_dir = os.path.join(args.output_dir, args.task)
    print(f'>>> Output directory: {args.output_dir}')
    print(f'-----------------------------------------------------------------------------')
    
    task_dick[args.task](args, load_model())