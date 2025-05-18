import os
import torch
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from PIL import Image

# Define the upscaling function
def upscale_image(input_path, output_path):
    model_path = 'Real-ESRGAN/experiments/pretrained_models/RealESRGAN_x4plus.pth'

    # Load the ESRGAN model
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64,
        num_block=23, num_grow_ch=32, scale=4
    )

    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=torch.cuda.is_available()
    )

    # Load and upscale the image
    img = Image.open(input_path).convert('RGB')
    img_np = np.array(img)

    try:
        output, _ = upsampler.enhance(img_np)
        Image.fromarray(output).save(output_path)
        print(f"Upscaled image saved to: {output_path}")
    except Exception as e:
        print(f"Upscaling failed: {e}")
