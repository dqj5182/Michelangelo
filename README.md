# Michelangelo

## [Conditional 3D Shape Generation based on Shape-Image-Text Aligned Latent Representation](https://neuralcarver.github.io/michelangelo)<br/>
https://github.com/NeuralCarver/Michelangelo/assets/37449470/123bae2c-fbb1-4d63-bd13-0e300a550868

Visualization of the 3D shape produced by our framework, which splits into triplets with a conditional input on the left, a normal map in the middle, and a triangle mesh on the right. The generated 3D shapes semantically conform to the visual or textural conditional inputs.<br/>

## 🔆 Features
**Michelangelo** possesses three capabilities: 

1. Representing a shape into shape-image-text aligned space;
2. Image-conditioned Shape Generation;
3. Text-conditioned Shape Generation.

## ⚙️ Setup

### Installation
Follow the command below to install the environment. We have tested the installation package on Tesla V100 and Tesla T4. 
```
git clone https://github.com/NeuralCarver/Michelangelo.git
cd Michelangelo
conda create --name Michelangelo python=3.9
conda activate Michelangelo 
pip install -r requirements.txt
```

### Checkpoints
Pleasae download weights from <a href="https://huggingface.co/Maikou/Michelangelo/tree/main/checkpoints">Hugging Face Model Space</a> and put it to root folder. We have also uploaded the weights related to CLIP to facilitate quick usage.

<details>
  <summary><b>  
    Tips for debugging configureation
  </b></summary>

- If something goes wrong in the environment configuration process unfortunately, the user may consider skipping those packages, such as pysdf, torch-cluster, and torch-scatter. These packages will not affect the execution of the commands we provide.
- If you encounter any issues while downloading CLIP, you can consider downloading it from [CLIP's Hugging Face page](https://huggingface.co/openai/clip-vit-large-patch14). Once the download is complete, remember to modify line [26](https://github.com/NeuralCarver/Michelangelo/blob/b53fa004cd4aeb0f4eb4d159ecec8489a4450dab/configs/text_cond_diffuser_asl/text-ASLDM-256.yaml#L26C1-L26C76) and line [34](https://github.com/NeuralCarver/Michelangelo/blob/b53fa004cd4aeb0f4eb4d159ecec8489a4450dab/configs/text_cond_diffuser_asl/text-ASLDM-256.yaml#L34) in the config file for providing correct path of CLIP.
- From [issue 6](https://github.com/NeuralCarver/Michelangelo/issues/6#issuecomment-1913513382). For Windows users, running wsl2 + ubuntu 22.04, will have issues. As discussed in [issue 786](https://github.com/microsoft/WSL/issues/8587) it is just a matter to add this in the .bashrc:
```
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH.
```
</details>

## ⚡ Quick Start

### Inference
#### Image-conditioned shape generation
```

CUDA_VISIBLE_DEVICES=4 python inference.py --task image2mesh --config_path ./configs/image_cond_diffuser_asl/image-ASLDM-256.yaml --ckpt_path ./checkpoints/image_cond_diffuser_asl/image-ASLDM-256.ckpt --image_path ./example_data/image/car.jpg
```