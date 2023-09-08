# Test Code

from idavatar.utils import parse_args
from idavatar.model_sd21 import IDAvatarCLIPImageEncoder, IDAvatarTextEncoder, IDAvatarModel
import torch
from diffusers import AutoencoderKL, DDPMScheduler
from idavatar.unet_21 import UNetModel

# correct

if __name__ == '__main__':
    from idavatar.utils import parse_args 
    sd_path = 'D:/apps/tools/programming/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1/snapshots/5cae40e6a2745ae2b01ad92ae5043f95f23644d6'
    clip_path = 'D:/apps/tools/programming/.cache/huggingface/hub/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/snapshots/94a64189c3535c1cb44acfcccd7b0908c1c8eb23/open_clip_pytorch_model.bin'
    
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
    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
    )

    idavatar(batch, noise_scheduler)