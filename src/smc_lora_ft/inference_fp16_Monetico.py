import os
import sys
sys.path.append("./")
from datetime import datetime
import time

import torch
from src.smc.transformer import Transformer2DModel
from src.smc_lora_ft.pipeline import Pipeline
from src.scheduler import Scheduler
from src.smc.scheduler import ReMDMScheduler, MeissonicScheduler
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)
from peft import LoraConfig
from diffusers.training_utils import cast_training_params
from diffusers.models.autoencoders.vq_model import VQModel
import src.smc.rewards as rewards
from src.smc.resampling import resample
from src.utils.metadata import get_metadata, save_metadata_json
    

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

use_remdm = False
if use_remdm:
    remdm_schedule = "cosine"
    remdm_remask_strategy = "rescale"
    remdm_eta = 0.05
    scheduler_new = ReMDMScheduler(
        schedule=remdm_schedule,
        remask_strategy=remdm_remask_strategy,
        eta=remdm_eta,
        mask_token_id=scheduler.config.mask_token_id, # type: ignore
    )
else:
    scheduler_new = MeissonicScheduler(
        mask_token_id=scheduler.config.mask_token_id, # type: ignore
        masking_schedule=scheduler.config.masking_schedule, #  type: ignore
    )
pipe = Pipeline(vq_model, tokenizer=tokenizer, text_encoder=text_encoder, transformer=model, scheduler=scheduler_new)
pipe.to(device)

# LORA lora checkpoint
lora_ckpt_uuid = 'a1e906e1-16a9-44a3-abe8-6dd2c17e12a2'
# lora_ckpt_uuid = ''
if lora_ckpt_uuid:
    ckpt_path = os.path.join('src/smc_lora_ft/checkpoints', lora_ckpt_uuid)
    pipe.load_lora_weights(
        pretrained_model_name_or_path_or_dict=ckpt_path,
    )

steps = 100
CFG = 9
resolution = 512 
negative_prompt = "worst quality, low quality, low res, blurry, distortion, watermark, logo, signature, text, jpeg artifacts, signature, sketch, duplicate, ugly, identifying mark"

prompts = [
    # "a photo of a bench left of a bear",
    # "a photo of a red orange and a purple broccoli",
    # "a photo of a brown giraffe and a white stop sign",
    # "a photo of a red dog",
    # "a photo of a white sandwich",
    # "a photo of four giraffes",
    # "a green stop sign in a red field",
    # "a photo of a green tennis racket and a black dog",
    "a photo of a yellow bird and a black motorcycle",
    # "a photo of an orange cow and a purple sandwich",
    # "a photo of a blue clock and a white cup",
    # "a photo of a brown knife and a blue donut",
    # ("A high-resolution image of a rabbit.", "A green colored rabbit"),
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

num_images = 8
batch_p = 1
# kl_weight = 0.0001 # pick score
# kl_weight = 0.01 # aesthetic score
kl_weight = 0.02 # image reward
# kl_weight = 10000
# proposal_type = "locally_optimal"
# proposal_type = "reverse"
proposal_type = "without_SMC"
# proposal_type = "straight_through_gradients"
resample_frequency = 10
partial_resampling = True
ess_threshold = 0.5
continuous_formulation = True

phi = 3
tau = 1.0
lambda_tempering = True
if lambda_tempering:
    lambda_one_at = min(100, steps)
    lambdas = torch.cat([torch.linspace(0, 1, lambda_one_at + 1), torch.ones(steps - lambda_one_at)])
else:
    lambdas = None

if isinstance(prompts[0], str):
    prompt = reward_prompt = prompts[0]
else:
    prompt, reward_prompt = prompts[0] # type: ignore
    

# reward_fn, reward_name = rewards.PickScore(device = 'cuda'), "pick"
# reward_fn, reward_name = rewards.aesthetic_score(device = 'cuda'), "aesthetic"
reward_fn, reward_name = rewards.ImageReward_Fk_Steering(device = 'cuda', bias=5.0), "image_reward_plus_5"
# reward_fn, reward_name = lambda images, prompts: rewards.color_match_reward(images, torch.tensor([1, 0, 0])), "color_red"
image_reward_fn = lambda images: reward_fn(
    images, 
    [reward_prompt] * len(images)
)

metadata = get_metadata(dict(locals()))

start_time = time.time()
images = pipe(
    prompt=prompt, 
    reward_fn=image_reward_fn,
    resample_fn=lambda log_w: resample(log_w, ess_threshold=ess_threshold, partial=partial_resampling),
    resample_frequency=resample_frequency,
    negative_prompt=negative_prompt,
    height=resolution,
    width=resolution,
    guidance_scale=CFG,
    num_inference_steps=steps,
    kl_weight=kl_weight,
    lambdas=lambdas,
    num_particles=num_images,
    batch_p=batch_p,
    proposal_type=proposal_type,
    use_continuous_formulation=continuous_formulation,
    phi=phi,
    tau=tau,
    output_type="pt",
)
end_time = time.time()
metadata["inference_time_seconds"] = end_time - start_time
print(f"Inference done in {metadata["inference_time_seconds"]:.2f} seconds")
metadata["gpu_memory_gb"] = torch.cuda.max_memory_allocated() / 1024**3
print("Max memory usage:", metadata["gpu_memory_gb"], "GB")


image_rewards = image_reward_fn(images)
pil_images = pipe.image_processor.postprocess(images, "pil") # type: ignore

save_best_image_only = False
best_image_reward, best_image_idx = image_rewards.max(dim=0)
print("Best image reward:", best_image_reward)

metadata["rewards"] = image_rewards.tolist()
metadata["best_reward"] = best_image_reward.item()
metadata["best_reward_idx"] = best_image_idx.item()

cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
output_dir = os.path.join("./output_SMC_lora_ft", cur_time)
os.makedirs(output_dir, exist_ok=True)
for i in range(len(images)):
    if save_best_image_only and i != best_image_idx:
        continue
    sanitized_prompt = prompt.replace(" ", "_")
    file_path = os.path.join(output_dir, f"{str(i).zfill(5)}.png")
    pil_images[i].save(file_path) #type: ignore
    print(f"The {i+1}/{num_images} image is saved to {file_path}")

save_metadata_json(metadata, output_dir)

# grid plot
import matplotlib.pyplot as plt
ncols = 4
nrows = (num_images + ncols - 1) // ncols
fig, ax = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
for i in range(len(images)):
    r = i // ncols
    c = i % ncols
    ax[r, c].imshow(pil_images[i]) #type: ignore
    ax[r, c].set_title(f"Reward: {image_rewards[i]:.2f}")
    ax[r, c].axis('off')
plt.savefig(os.path.join(output_dir, "grid.png"), bbox_inches='tight')
plt.close()
