import os
import torch
from torchvision.io import read_image, ImageReadMode
import glob
import json
import numpy as np
import random
from copy import deepcopy
import tqdm
from idavatar.datasets_utils.build.prompt_utils import generate_augment_prompt

class IDAvatarDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images_dir,
        masks_dir,
        prompts_path,
        tokenizer,
        train_transforms,
        object_transforms,
        object_processor,
        device=None,
        image_token="<|image|>",
        coarse_person = ['man', 'woman', 'guy', 'boy', 'girl', 'lady', 'sir', 'male', 'female', 'person', 'student'],
    ):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.prompts_path = prompts_path
        self.tokenizer = tokenizer
        self.train_transforms = train_transforms
        self.object_transforms = object_transforms
        self.object_processor = object_processor
        self.image_token = image_token
        self.device = device

        self.image_ids = os.listdir(images_dir)
        self.coarse_person = coarse_person

        tokenizer.add_tokens([image_token], special_tokens=True)
        self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)
        with open(self.prompts_path, "r") as f:
            self.prompts_json = json.load(f)


    def __len__(self):
        return len(self.image_ids)

    def _tokenize_and_mask_noun_phrases_ends(self, caption):
        input_ids = self.tokenizer.encode(caption)
        noun_phrase_end_mask = [False for _ in input_ids]
        clean_input_ids = []
        clean_index = 0

        for i, id in enumerate(input_ids):
            if id == self.image_token_id:
                noun_phrase_end_mask[clean_index - 1] = True
            else:
                clean_input_ids.append(id)
                clean_index += 1

        max_len = self.tokenizer.model_max_length

        if len(clean_input_ids) > max_len:
            clean_input_ids = clean_input_ids[:max_len]
        else:
            clean_input_ids = clean_input_ids + [self.tokenizer.pad_token_id] * (
                max_len - len(clean_input_ids)
            )

        if len(noun_phrase_end_mask) > max_len:
            noun_phrase_end_mask = noun_phrase_end_mask[:max_len]
        else:
            noun_phrase_end_mask = noun_phrase_end_mask + [False] * (
                max_len - len(noun_phrase_end_mask)
            )

        clean_input_ids = torch.tensor(clean_input_ids, dtype=torch.long)
        noun_phrase_end_mask = torch.tensor(noun_phrase_end_mask, dtype=torch.bool)
        return clean_input_ids.unsqueeze(0), noun_phrase_end_mask.unsqueeze(0)


    @torch.no_grad()
    def preprocess(self, image, mask, prompt):
        caption = generate_augment_prompt(
                self.coarse_person,
                prompt,
                self.special_token            
            )
        segment = mask

        pixel_value, transformed_segmap = self.train_transforms(image, segment)

        # 如果segmap已经被处理好了，则不需要这个
        # person_pixel_value = torch.where(transformed_segmap==1, pixel_value, 0)
        # 默认mask已经处理好了
        person_pixel_value = transformed_segmap

        input_ids, image_token_mask = self._tokenize_and_mask_noun_phrases_ends(
            caption=caption
        ) # 去除<|image|>后的input_ids (<|image|>在人的模糊表达后面)，对应的man或者woman为True，其余为False的mask

        return {
            "pixel_values": pixel_value,
            "person_pixel_values": person_pixel_value,
            "input_ids": input_ids,
        }

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        image_path = os.path.join(self.images_dir, image_id)
        mask_path = os.path.join(self.masks_dir, image_id)

        image = read_image(image_path, mode=ImageReadMode.RGB)
        mask = read_image(mask_path, mode=ImageReadMode.RGB)

        # TODO prompt以什么形式保存还不知道，io次数过长，是不是要直接存在json里
        # with open(self.prompts_path, "r") as f:
        #     prompt = json.load(f)[img_key]

        prompt = self.prompts_json[image_id]

        if self.device is not None:
            image = image.to(self.device)
            segmap = segmap.to(self.device)

        return self.preprocess(image, mask, prompt)


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    input_ids = torch.cat([example["input_ids"] for example in examples])

    image_token_mask = torch.cat([example["image_token_mask"] for example in examples])

    person_pixel_values = torch.stack(
        [example["person_pixel_values"] for example in examples]
    )

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "image_token_mask": image_token_mask,
        "person_pixel_values": person_pixel_values
    }


def get_data_loader(dataset, batch_size, shuffle=True):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=0,
    )

    return dataloader

