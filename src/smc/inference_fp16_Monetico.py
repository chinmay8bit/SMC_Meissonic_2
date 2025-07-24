import os
import sys
sys.path.append("./")

import torch
from torchvision import transforms
from src.smc.transformer import Transformer2DModel
from src.smc.pipeline import Pipeline
from src.scheduler import Scheduler
from src.smc.scheduler import ReMDMScheduler
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)
from diffusers.models.autoencoders.vq_model import VQModel
import src.smc.rewards as rewards
from src.smc.resampling import resample

device = 'cuda'
dtype = torch.bfloat16
model_path = "Collov-Labs/Monetico"
model = Transformer2DModel.from_pretrained(model_path, subfolder="transformer", torch_dtype=dtype)
vq_model = VQModel.from_pretrained(model_path, subfolder="vqvae", torch_dtype=dtype)
text_encoder = CLIPTextModelWithProjection.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=dtype) # better for Monetico
# text_encoder = CLIPTextModelWithProjection.from_pretrained(  #more stable sampling for some cases
#             "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", torch_dtype=dtype
#         )
tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", torch_dtype=dtype)
scheduler = Scheduler.from_pretrained(model_path, subfolder="scheduler", torch_dtype=dtype)
scheduler_new = ReMDMScheduler(
    schedule="cosine",
    remask_strategy="rescale",
    eta=0.1,
    mask_token_id=scheduler.config.mask_token_id, # type: ignore
)
pipe = Pipeline(vq_model, tokenizer=tokenizer, text_encoder=text_encoder, transformer=model, scheduler=scheduler_new, device=device, model_dtype=dtype)

steps = 48
CFG = 9
resolution = 512 
negative_prompt = "worst quality, low quality, low res, blurry, distortion, watermark, logo, signature, text, jpeg artifacts, signature, sketch, duplicate, ugly, identifying mark"

prompts = [
    # "Three boats in the ocean with a rainbow in the sky.", 
    # "Two actors are posing for a pictur with one wearing a black and white face paint.",
    "A large body of water with a rock in the middle and mountains in the background.",
    "A white and blue coffee mug with a picture of a man on it.",
    "A statue of a man with a crown on his head.",
    "A man in a yellow wet suit is holding a big black dog in the water.",
    "A white table with a vase of flowers and a cup of coffee on top of it.",
    "A woman stands on a dock in the fog.",
    "A woman is standing next to a picture of another woman."
]

num_images = 4
batch_p = 1
kl_weight = 0.0005
# kl_weight = 10000

if isinstance(prompts[0], str):
    prompt = reward_prompt = prompts[0]
else:
    prompt, reward_prompt = prompts[0] # type: ignore
    

reward_fn, reward_name = rewards.PickScore(device = 'cuda'), "pick"
# reward_fn, reward_name = rewards.aesthetic_score(device = 'cuda'), "aesthetic"
# reward_fn, reward_name = lambda images, prompts: rewards.color_match_reward(images, torch.tensor([1, 0, 0])), "color_red"
image_reward_fn = lambda images: reward_fn(
    images, 
    [reward_prompt] * len(images)
)

images = pipe(
    prompt=prompt, 
    reward_fn=image_reward_fn,
    resample_fn=lambda log_w: resample(log_w, ess_threshold=0.5, partial=False),
    negative_prompt=negative_prompt,
    height=resolution,
    width=resolution,
    guidance_scale=CFG,
    num_inference_steps=steps,
    kl_weight=kl_weight,
    num_particles=num_images,
    batch_p=batch_p,
)

output_dir = "./output_SMC"
os.makedirs(output_dir, exist_ok=True)
for i in range(len(images)):
    sanitized_prompt = prompt.replace(" ", "_")
    file_path = os.path.join(output_dir, f"{sanitized_prompt}_{resolution}_{steps}_{CFG}.png")
    images[i].save(file_path) #type: ignore
    print(f"The {i+1}/{num_images} image is saved to {file_path}")
