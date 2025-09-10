# primary generation script
import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt

import argparse

from datetime import datetime

import torch
from src.smc.transformer import Transformer2DModel
from src.smc.pipeline import Pipeline
from src.scheduler import Scheduler
from src.smc.scheduler import ReMDMScheduler, MeissonicScheduler
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)
from diffusers.models.autoencoders.vq_model import VQModel

from src.smc.resampling import resample
import src.smc.rewards as rewards
from src.eval.fks_utils import do_eval


# load prompt data
def load_geneval_metadata(prompt_path, max_prompts=None):
    prompt_path = "src/eval/prompt_files/" + prompt_path
    if prompt_path.endswith(".json"):
        with open(prompt_path, "r") as f:
            data = json.load(f)
    else:
        assert prompt_path.endswith(".jsonl")
        with open(prompt_path, "r") as f:
            data = [json.loads(line) for line in f]
    assert isinstance(data, list)
    prompt_key = "prompt"
    if prompt_key not in data[0]:
        assert "text" in data[0], "Prompt data should have 'prompt' or 'text' key"

        for item in data:
            item["prompt"] = item["text"]
    if max_prompts is not None:
        data = data[:max_prompts]
    return data


def get_model(args, device):
    if args.model_name == "meissonic":
        model_path = "MeissonFlow/Meissonic"
        dtype = torch.float
    elif args.model_name == "meissonic-fp16-monetico":
        model_path = "Collov-Labs/Monetico"
        dtype = torch.bfloat16
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")
    model = Transformer2DModel.from_pretrained(model_path, subfolder="transformer", torch_dtype=dtype)
    vq_model = VQModel.from_pretrained(model_path, subfolder="vqvae", torch_dtype=dtype)
    text_encoder = CLIPTextModelWithProjection.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=dtype) # better for Monetico
    # text_encoder = CLIPTextModelWithProjection.from_pretrained(  #more stable sampling for some cases
    #             "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", torch_dtype=dtype
    #         )
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", torch_dtype=dtype)
    scheduler = Scheduler.from_pretrained(model_path, subfolder="scheduler", torch_dtype=dtype)

    if args.use_remdm:
        scheduler_new = ReMDMScheduler(
            schedule=args.remdm_schedule,
            remask_strategy=args.remdm_remask_strategy,
            eta=args.remdm_eta,
            mask_token_id=scheduler.config.mask_token_id, # type: ignore
        )
    else:
        scheduler_new = MeissonicScheduler(
            mask_token_id=scheduler.config.mask_token_id, # type: ignore
            masking_schedule=scheduler.config.masking_schedule, #  type: ignore
        )
    pipe = Pipeline(vq_model, tokenizer=tokenizer, text_encoder=text_encoder, transformer=model, scheduler=scheduler_new, device=device, model_dtype=dtype)
    return pipe



def main(args):
    # seed everything
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load prompt data
    prompt_data = load_geneval_metadata(args.prompt_path)

    # configure pipeline
    pipe = get_model(args, device)
    
    if args.reward_name == "aesthetic":
        reward_fn = rewards.aesthetic_score(device = 'cuda')
    elif args.reward_name == "image_reward":
        reward_fn = rewards.ImageReward_Fk_Steering(device = 'cuda')
    elif args.reward_name == "image_reward_plus_5":
        reward_fn = rewards.ImageReward_Fk_Steering(device = 'cuda', bias=5.0)
    elif args.reward_name == "pick":
        reward_fn = rewards.PickScore(device = 'cuda')
    else:
        raise ValueError("Invalid reward name")
    
    if args.lambda_tempering:
        lambdas = torch.cat([torch.linspace(0, 1, args.lambda_one_at + 1), torch.ones(args.num_inference_steps - args.lambda_one_at)])
    else:
        lambdas = None

    # set output directory
    cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(args.output_dir, cur_time)
    try:
        os.makedirs(output_dir, exist_ok=False)
    except FileExistsError:
        # make file sleep for a random time
        import time

        print("Sleeping for a random time")

        time.sleep(np.random.randint(1, 10))

        cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = os.path.join(args.output_dir, cur_time)
        os.makedirs(output_dir, exist_ok=False)

    arg_path = os.path.join(output_dir, "args.json")
    with open(arg_path, "w") as f:
        json.dump(vars(args), f, indent=4)

    metrics_to_compute = args.metrics_to_compute.split("#")
    
    # cache metric fns
    do_eval(
        prompt=["test"],
        images=[Image.new("RGB", (224, 224))],
        metrics_to_compute=metrics_to_compute,
    )

    metrics_arr = {
        metric: dict(mean=0.0, max=0.0, min=0.0, std=0.0) for metric in metrics_to_compute
    }
    n_samples = 0
    average_time = 0

    for prompt_idx, item in enumerate(tqdm(prompt_data)):
        prompt = item["prompt"]
        prompt_arr = [prompt] * args.num_particles
        start_time = datetime.now()

        prompt_path = os.path.join(output_dir, f"{prompt_idx:0>5}")
        os.makedirs(prompt_path, exist_ok=True)

        # dump metadata
        with open(os.path.join(prompt_path, "metadata.jsonl"), "w") as f:
            json.dump(item, f)
            
        image_reward_fn = lambda images: reward_fn(
            images, 
            [prompt] * len(images)
        )

        img_size = 1024 if args.model_name == "meissonic" else 512
        negative_prompt = "worst quality, low quality, low res, blurry, distortion, watermark, logo, signature, text, jpeg artifacts, signature, sketch, duplicate, ugly, identifying mark"
        images = pipe(
            prompt=prompt, 
            reward_fn=image_reward_fn,
            resample_fn=lambda log_w: resample(log_w, ess_threshold=args.ess_threshold, partial=args.partial_resampling),
            resample_frequency=args.resample_frequency,
            negative_prompt=negative_prompt,
            height=img_size,
            width=img_size,
            guidance_scale=args.CFG,
            num_inference_steps=args.num_inference_steps,
            kl_weight=args.kl_weight,
            lambdas=lambdas,
            num_particles=args.num_particles,
            batch_p=args.batch_p,
            proposal_type=args.proposal_type,
            use_continuous_formulation=args.continuous_formulation,
            phi=args.phi,
            tau=args.tau,
            output_type="pil",
        )
        if not args.proposal_type == "without_SMC":
            end_time = datetime.now()

        results = do_eval(
            prompt=prompt_arr, images=images, metrics_to_compute=metrics_to_compute
        )
        if args.proposal_type == "without_SMC":
            end_time = datetime.now()
        time_taken = end_time - start_time #type: ignore

        results["time_taken"] = time_taken.total_seconds()
        results["prompt"] = prompt
        results["prompt_index"] = prompt_idx

        n_samples += 1

        average_time += time_taken.total_seconds()
        print(f"Time taken: {average_time / n_samples}")
        
        print("Max GPU memory used:", torch.cuda.max_memory_allocated() / 1024**3, "GB")

        # sort images by reward
        guidance_reward = np.array(results[args.guidance_reward_fn]["result"])
        sorted_idx = np.argsort(guidance_reward)[::-1]
        images = [images[i] for i in sorted_idx]
        for metric in metrics_to_compute:
            results[metric]["result"] = [
                results[metric]["result"][i] for i in sorted_idx
            ]

        for metric in metrics_to_compute:
            metrics_arr[metric]["mean"] += results[metric]["mean"]
            metrics_arr[metric]["max"] += results[metric]["max"]
            metrics_arr[metric]["min"] += results[metric]["min"]
            metrics_arr[metric]["std"] += results[metric]["std"]

        for metric in metrics_to_compute:
            print(
                metric,
                metrics_arr[metric]["mean"] / n_samples,
                metrics_arr[metric]["max"] / n_samples,
            )

        if args.save_individual_images:
            sample_path = os.path.join(prompt_path, "samples")
            os.makedirs(sample_path, exist_ok=True)
            for image_idx, image in enumerate(images):
                image.save(os.path.join(sample_path, f"{image_idx:05}.png")) # type: ignore

            best_of_n_sample_path = os.path.join(prompt_path, "best_of_n_samples")
            os.makedirs(best_of_n_sample_path, exist_ok=True)
            for image_idx, image in enumerate(images[:1]):
                image.save(os.path.join(best_of_n_sample_path, f"{image_idx:05}.png")) # type: ignore

        with open(os.path.join(prompt_path, "results.json"), "w") as f:
            json.dump(results, f)

        _, ax = plt.subplots(1, args.num_particles, figsize=(args.num_particles * 5, 5))
        if args.num_particles == 1:
            ax = [ax]
        for i, image in enumerate(images):
            ax[i].imshow(image) # type: ignore
            ax[i].axis("off") # type: ignore

        plt.suptitle(prompt[0])
        image_fpath = os.path.join(prompt_path, f"grid.png")
        plt.savefig(image_fpath)
        plt.close()

    # save final metrics
    for metric in metrics_to_compute:
        metrics_arr[metric]["mean"] /= n_samples
        metrics_arr[metric]["max"] /= n_samples
        metrics_arr[metric]["min"] /= n_samples
        metrics_arr[metric]["std"] /= n_samples

    with open(os.path.join(output_dir, "final_metrics.json"), "w") as f:
        json.dump(metrics_arr, f)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="geneval_outputs")
    parser.add_argument("--save_individual_images", type=bool, default=True)
    parser.add_argument("--num_particles", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=100)
    parser.add_argument(
        "--metrics_to_compute",
        type=str,
        default="ImageReward#HumanPreference",
        help="# separated list of metrics",
    )
    parser.add_argument("--guidance_reward_fn", type=str, default="ImageReward")
    parser.add_argument("--prompt_path", type=str, default="geneval_metadata.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resample_frequency", type=int, default=5)
    parser.add_argument("--reward_name", type=str, default="image_reward_plus_5")
    parser.add_argument("--lambda_tempering", action="store_true")
    parser.add_argument("--model_name", type=str, default="meissonic")
    parser.add_argument("--use_remdm", action="store_true")
    parser.add_argument("--CFG", type=float, default=9.0)
    parser.add_argument("--proposal_type", type=str, default="locally_optimal")
    parser.add_argument("--phi", type=int, default=1)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--ess_threshold", type=float, default=0.5)
    parser.add_argument("--partial_resampling", action="store_true")
    parser.add_argument("--kl_weight", type=float, default=1.0)
    parser.add_argument("--lambda_one_at", type=int, default=100)
    parser.add_argument("--continuous_formulation", action="store_true")
    parser.add_argument("--batch_p", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.prompt_path == "geneval_metadata.jsonl":
        args.save_individual_images = True

    args.output_dir = args.prompt_path.replace(".jsonl", f"_outputs").replace(".json", f"_outputs")

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
