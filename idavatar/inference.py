from idavatar.transforms import get_object_transforms
from idavatar.data import DemoDataset #TODO
from idavatar.model import IDAvatarModel
from diffusers import PNDMScheduler
from transformers import CLIPTokenizer
from accelerate.utils import set_seed
from idavatar.utils import parse_args
from accelerate import Accelerator
from pathlib import Path
from collections import OrderedDict
from idavatar.transforms import PadToSquare
from torchvision import transforms as T
from PIL import Image
import mmcv
import numpy as np
import torch
import os
from tqdm.auto import tqdm
import types
import itertools
import os
import yaml


args = parse_args()

sd_weights_path = 'D:/apps/tools/programming/.cache/huggingface/stable-diffusion-v1-5'

transforms = torch.nn.Sequential(
            OrderedDict(
                [
                    ("pad_to_square", PadToSquare(fill=0, padding_mode="constant")),
                    (
                        "resize",
                        T.Resize(
                            (256, 256),
                            interpolation=T.InterpolationMode.BILINEAR,
                            antialias=True,
                        ),
                    ),
                    ("convert_to_float", T.ConvertImageDtype(torch.float32)),
                ]
            )
        )

image_path = './images'
caption = 'a man img running on the road'

image_list = []

for image in os.listdir(image_path):
    image = mmcv.imread(os.path.join(image_path, image))
    image = torch.tensor(image).permute(2, 0, 1)
    image = transforms(image)
    image_list.append(image)

# num_images, c, h, w
images = torch.stack(image_list, dim=0)

tokenizer = CLIPTokenizer.from_pretrained(sd_weights_path, subfolder='tokenizer')
scheduler = PNDMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                set_alpha_to_one=False,
                skip_prk_steps=True,
            )

args.fuser_scale=0.3
model = IDAvatarModel.from_pretrained(args)

image = model.generate(
                        images,
                        caption,
                        'img',
                        tokenizer,
                        scheduler
                    )