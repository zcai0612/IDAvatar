# Test Code

from idavatar.utils import parse_args
from idavatar.model import IDAvatarCLIPImageEncoder, IDAvatarTextEncoder, IDAvatarModel
import torch
from diffusers import AutoencoderKL, DDPMScheduler
from idavatar.unet import UNetModel

# correct

if __name__ == '__main__':
    from idavatar.utils import parse_args 
    sd_path = 'D:/apps/tools/programming/.cache/huggingface/hub/models--SG161222--Realistic_Vision_V5.1_noVAE/snapshots/9cd4afd23ecbf0348e2c46f4ac712dbf032da73c'
    clip_path = 'D:/apps/tools/programming/.cache/huggingface/hub/models--laion--CLIP-ViT-L-14-DataComp.XL-s13B-b90K/snapshots/84c9828e63dc9a9351d1fe637c346d4c1c4db341/open_clip_pytorch_model.bin'
    
    text_encoder = IDAvatarTextEncoder.from_pretrained(sd_path, subfolder='text_encoder')
    image_encoder = IDAvatarCLIPImageEncoder.from_pretrained(clip_path)
    vae = AutoencoderKL.from_pretrained(sd_path, subfolder='vae')
    args = parse_args()

    unet = UNetModel()
    idavatar = IDAvatarModel(
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            vae=vae,
            unet=unet,
            controlnet=None,
            args=args,
        )
    batch = {}
    pixel_values = torch.rand((2, 3, 256, 256))
    input_ids = torch.randint(low=1, high=500, size=(2, 77))
    image_token_mask = torch.full_like(input_ids, False)
    image_token_mask[:, 2] = True
    object_pixel_values = torch.rand((2, 3, 256, 256))

    batch["pixel_values"] = pixel_values
    batch["input_ids"] = input_ids
    batch["image_token_mask"] = image_token_mask
    batch["object_pixel_values"] = object_pixel_values
    noise_scheduler = DDPMScheduler.from_pretrained(
        sd_path, subfolder="scheduler"
    )

    idavatar(batch, noise_scheduler)