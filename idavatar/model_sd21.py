import torch
import torch.nn as nn
from diffusers import AutoencoderKL, StableDiffusionPipeline, ControlNetModel
from idavatar.unet_21 import UNetModel
from transformers import CLIPTextModel
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import (
    _expand_mask,
    CLIPTextTransformer,
    CLIPPreTrainedModel,
    CLIPModel,
)
import tqdm
import types
import torchvision.transforms as T
import gc
import numpy as np
import open_clip

inference_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, use_residual=True):
        super().__init__()
        if use_residual:
            assert in_dim == out_dim
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.use_residual = use_residual
        self.act_fn = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.layernorm(x)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        if self.use_residual:
            x = x + residual
        return x


class IDAvatarCLIPImageEncoder(nn.Module):
    @staticmethod
    def from_pretrained(
        global_model_name_or_path,
    ):
        model,_,_ = open_clip.create_model_and_transforms('ViT-H-14', pretrained=global_model_name_or_path)
        vision_model = model.visual
        vision_model.output_tokens = True
        vision_processor = T.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        )
        return IDAvatarCLIPImageEncoder(
            vision_model,
            vision_processor,
        )

    def __init__(
        self,
        vision_model,
        vision_processor,
    ):
        super().__init__()
        self.vision_model = vision_model
        self.vision_processor = vision_processor

        self.image_size = vision_model.image_size

    def forward(self, person_pixel_values):
        b, c, h, w = person_pixel_values.shape

        if (h, w) != self.image_size:
            h, w = self.image_size
            person_pixel_values = F.interpolate(
                person_pixel_values, (h, w), mode="bilinear", antialias=True
            )# b, c, h, w -> b, c, 224, 224
        person_pixel_values = self.vision_processor(person_pixel_values) 
        person_embeds, patch_features = self.vision_model(person_pixel_values)# b, 1024; b, 256, 1280
        person_embeds = person_embeds.view(b, 1, -1) # b, 1, 1024
        return person_embeds, patch_features


def fuse_object_embeddings(
    inputs_embeds, # b, 77, 1024
    image_token_mask, # b, 77
    object_embeds, # b, 1, 1024
    alpha=1,
):
    object_embeds = object_embeds.to(inputs_embeds.dtype)

    batch_size, obj_seq_length = object_embeds.shape[:2]
    seq_length = inputs_embeds.shape[1]

    inputs_embeds = inputs_embeds.view(-1, inputs_embeds.shape[-1]) # b*77, 1024
    image_token_mask = image_token_mask.view(-1) # b*77
    valid_object_embeds = object_embeds.view(-1, object_embeds.shape[-1]) # b*num_objs, 1024

    # slice out the image token embeddings
    # image_token_embeds = inputs_embeds[image_token_mask] # b*77, 1024 chose the only one image_token_embeds - img
    # valid_object_embeds = alpha*valid_object_embeds + (1-alpha)*image_token_embeds # let subject image token embeddings fused with image embeddings

    orignal_input_embeds = inputs_embeds
    inputs_embeds.masked_scatter_(image_token_mask[:, None], valid_object_embeds) # replace
    outputs_embeds = alpha*inputs_embeds + (1-alpha) * orignal_input_embeds
    outputs_embeds = outputs_embeds.view(batch_size, seq_length, -1) # bsz, 77, 1024
    return outputs_embeds


def post_fuse(
    text_embeds,
    object_embeds,
    image_token_mask,
    alpha=1,
) -> torch.Tensor:
    text_object_embeds = fuse_object_embeddings(
        text_embeds, image_token_mask, object_embeds, alpha # balance the weights of identiyt image when inference
    )

    return text_object_embeds


class IDAvatarTextEncoder(CLIPPreTrainedModel):
    _build_causal_attention_mask = CLIPTextTransformer._build_causal_attention_mask

    @staticmethod
    def from_pretrained(model_name_or_path, **kwargs):
        model = CLIPTextModel.from_pretrained(model_name_or_path, **kwargs)
        text_model = model.text_model
        return IDAvatarTextEncoder(text_model)

    def __init__(self, text_model):
        super().__init__(text_model.config)
        self.config = text_model.config
        self.final_layer_norm = text_model.final_layer_norm
        self.embeddings = text_model.embeddings
        self.encoder = text_model.encoder

    def forward(
        self,
        input_ids,
        use_causual_mask=True,
    ):

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids)

        bsz, seq_len = input_shape

        if use_causual_mask:
            causal_attention_mask = self._build_causal_attention_mask(
                                        bsz, seq_len, hidden_states.dtype
                                    ).to(self.device)
            
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            causal_attention_mask=causal_attention_mask,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(
                dim=-1
            ),
        ]

        return (last_hidden_state, pooled_output) + encoder_outputs[1:]

# input: bsz, 1, 1024
# output: bsz, 1, 1024
class ConceptMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = MLP(in_dim, out_dim, out_dim, use_residual=True)
        self.layer_norm = nn.LayerNorm(out_dim)
    
    def forward(self, object_embeds):
        object_embeds = self.mlp(object_embeds)
        object_embeds = self.layer_norm(object_embeds)

        return object_embeds
        
# input: bsz, 256, 1024
# output: bsz, 256, 1024
class PatchMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = MLP(in_dim, out_dim, out_dim, use_residual=False)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, patch_features):
        patch_features = self.mlp(patch_features)
        patch_features = self.layer_norm(patch_features)

        return patch_features

class IDAvatarModel(nn.Module):
    def __init__(self, text_encoder, image_encoder, vae, unet, controlnet, args):
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.vae = vae
        self.unet = unet
        self.controlnet = controlnet
        self.use_ema = False
        self.ema_param = None
        self.pretrained_model_name_or_path = args.pretrained_model_name_or_path
        self.revision = args.revision
        embed_dim = text_encoder.config.hidden_size
        self.concept_mlp = ConceptMLP(embed_dim, embed_dim) 
        self.patch_mlp = PatchMLP(1280, embed_dim) # TODO How to get the dim of patch features?
        # patch_features_out_dim = embed_dim

    @property
    def device(self):
        return list(self.parameters())[0].device

    def _init_latent(self, latent, height, width, generator, batch_size):
        if latent is None:
            latent = torch.randn(
                (1, self.unet.in_channels, height // 8, width // 8),
                generator=generator,
                device=self.device,
            )
        latent = latent.expand(
            batch_size,
            self.unet.in_channels,
            height // 8,
            width // 8,
        )
        return latent

    def _latent_to_image(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    @staticmethod
    def from_pretrained(args):
        text_encoder = IDAvatarTextEncoder.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=args.revision,
        )
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
        )
        unet = UNetModel()
        unet.load_state_dict(torch.load(args.unet_model_path), strict=False)

        image_encoder = IDAvatarCLIPImageEncoder.from_pretrained(
            args.image_encoder_name_or_path,
        )
        if args.controlnet_pretrained_name_or_path:
            controlnet = ControlNetModel.from_pretrained(
                args.controlnet_pretrained_name_or_path,
            )
        else:
            controlnet = None

        return IDAvatarModel(text_encoder, image_encoder, vae, unet, controlnet, args)


    def forward(self, batch, noise_scheduler):
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        image_token_mask = batch["image_token_mask"]
        object_pixel_values = batch["object_pixel_values"]

        vae_dtype = self.vae.parameters().__next__().dtype
        vae_input = pixel_values.to(vae_dtype)

        latents = self.vae.encode(vae_input).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, noise_scheduler.num_train_timesteps, (bsz,), device=self.device
        )
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # (bsz, num_image_tokens, dim) (bsz, 1, 1024) and (bsz, 256, 1280)
        object_embeds, patch_features = self.image_encoder(object_pixel_values)

        object_embeds = self.concept_mlp(object_embeds)

        patch_features = self.patch_mlp(patch_features)

        encoder_hidden_states = self.text_encoder(
            input_ids
        )[
            0
        ]  # (bsz, seq_len, dim), (bsz, 77, 1024)

        # (bsz, 77, 1024)
        encoder_hidden_states = post_fuse(
            encoder_hidden_states,
            object_embeds,
            image_token_mask,
        )
        
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {noise_scheduler.config.prediction_type}"
            )

        pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, patch_features)

        denoise_loss = F.mse_loss(pred.float(), target.float(), reduction="mean")

        return_dict = {"denoise_loss": denoise_loss}

        return return_dict


    def generate(
            self, 
            pixel_values,  # num_pics, c, h, w
            caption,
            image_token,
            tokenizer,
            scheduler,
            alpha_=0.7,
            height=512,
            width=512,
            seed=42,
            num_inference_steps=50,
            neg_prompt="",
            latent=None,
            guidance_scale=7.5,
        ):
        tokenizer.add_tokens([image_token], special_tokens=True)
        input_ids, image_token_mask = _tokenize_and_mask_noun_phrases_ends(
                                                                        caption, 
                                                                        tokenizer, 
                                                                        image_token
                                                                    )
        input_ids.to(self.device)
        image_token_mask.to(self.device)

        # num_pics, 1, 1024; num_pics, 256, 1280
        pixel_values_embeds, patch_features = self.image_encoder(pixel_values)
        # num_pics, 1, 1024 => 1, 1, 1024
        person_embeds = pixel_values_embeds.mean(dim=0, keepdim=True)
        patch_features = patch_features.mean(dim=0, keepdim=True)
        
        person_embeds = self.concept_mlp(person_embeds)
        patch_features = self.patch_mlp(patch_features)

        encoder_hidden_states = self.text_encoder(
            input_ids, image_token_mask, person_embeds
        )[0]  # (bsz, seq_len, dim)

        encoder_hidden_states = post_fuse(
            encoder_hidden_states,
            person_embeds,
            image_token_mask,
        )

        patch_features = self.patch_mlp(patch_features)

        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            max_length = tokenizer.model_max_length

            clear_caption = caption.replace(image_token+' ', '')
            text_only_input = tokenizer(
                [clear_caption],
                padding="max_length",
                max_length=max_length,
                return_tensor='pt'
            )

            text_only_embeddings = self.text_encoder(
                input_ids=text_only_input.input_ids.to(self.device)
            )[0]

            neg_input = tokenizer(
                [neg_prompt],
                padding="max_length",
                max_length=max_length,
                return_tensor='pt'
            )
            # b, 77, 768
            neg_embeddings = self.text_encoder(
                input_ids=neg_input.input_ids.to(self.device)
            )[0]
        else:
            assert 0, 'Not Finshed!'

        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator = generator.manual_seed(seed)

        latents = self._init_latent(latent, height, width, generator, batch_size=1)

        start_subject_conditioning_step = (1-alpha_) * num_inference_steps

        scheduler.set_timesteps(num_inference_steps)

        iterator = tqdm.tqdm(scheduler.timesteps)

        for i, t in enumerate(iterator):
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            patch_features = torch.cat([patch_features]*2, dim=0)

            if i <= start_subject_conditioning_step:
                current_prompt_embeds = torch.cat(
                    [neg_embeddings, text_only_embeddings], dim=0
                )

            else:
                current_prompt_embeds = torch.cat(
                    [neg_embeddings, encoder_hidden_states], dim=0
                )
            
            noise_pred = self.unet(
                latent_model_input, t, current_prompt_embeds, patch_features
            )
            if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
            else:
                assert 0, "Not Finished!"

            latents = scheduler.step(
                noise_pred,
                t,
                latents,
            )["prev_sample"]

        generated_image = self._latent_to_image(latents=latents)

        return generated_image


def _tokenize_and_mask_noun_phrases_ends(caption, tokenizer, image_token):
    image_token_id = tokenizer.convert_tokens_to_ids(image_token)
    input_ids = tokenizer.encode(caption)
    noun_phrase_end_mask = [False for _ in input_ids]
    clean_input_ids = []
    clean_index = 0

    for i, id in enumerate(input_ids):
        if id == image_token_id:
            noun_phrase_end_mask[clean_index - 1] = True
        else:
            clean_input_ids.append(id)
            clean_index += 1

    max_len = tokenizer.model_max_length

    if len(clean_input_ids) > max_len:
        clean_input_ids = clean_input_ids[:max_len]
    else:
        clean_input_ids = clean_input_ids + [tokenizer.pad_token_id] * (
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

