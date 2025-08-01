from PIL import Image
import torch
from importlib import resources
ASSETS_PATH = resources.files("assets")

def jpeg_compressibility(inference_dtype=None, device=None):
    import io
    import numpy as np
    def loss_fn(images):
        if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        loss = torch.tensor(sizes, dtype=inference_dtype, device=device)
        rewards = -1 * loss

        return loss, rewards

    return loss_fn

def clip_score(
    inference_dtype=None, 
    device=None, 
    return_loss=False, 
):
    from src.smc.scorers.clip_scorer import CLIPScorer

    scorer = CLIPScorer(dtype=torch.float32, device=device)
    scorer.requires_grad_(False)

    if not return_loss:
        def _fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)
            return scores

        return _fn

    else:
        def loss_fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)

            loss = - scores
            return loss, scores

        return loss_fn

def aesthetic_score(
    torch_dtype=None,
    aesthetic_target=None,
    grad_scale=0,
    device=None,
    return_loss=False,
):
    from src.smc.scorers.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32, device=device)
    scorer.requires_grad_(False)

    if not return_loss:
        def _fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images)
            return scores

        return _fn

    else:
        def loss_fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images)

            if aesthetic_target is None: # default maximization
                loss = -1 * scores
            else:
                # using L1 to keep on same scale
                loss = abs(scores - aesthetic_target)
            return loss * grad_scale, scores

        return loss_fn


def hps_score(
    inference_dtype=None, 
    device=None, 
    return_loss=False, 
):
    from src.smc.scorers.hpsv2_scorer import HPSv2Scorer

    scorer = HPSv2Scorer(dtype=torch.float32, device=device)
    scorer.requires_grad_(False)

    if not return_loss:
        def _fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)
            return scores

        return _fn

    else:
        def loss_fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)

            loss = 1.0 - scores
            return loss, scores

        return loss_fn


def ImageReward(
    inference_dtype=None, 
    device=None, 
    return_loss=False, 
):
    from src.smc.scorers.ImageReward_scorer import ImageRewardScorer

    scorer = ImageRewardScorer(dtype=torch.float32, device=device)
    scorer.requires_grad_(False)

    if not return_loss:
        def _fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)
            return scores

        return _fn

    else:
        def loss_fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)

            loss = - scores
            return loss, scores

        return loss_fn
    
    
def ImageReward_Fk_Steering(
    inference_dtype=None, 
    device=None, 
    return_loss=False, 
    bias=None,
):
    from src.smc.scorers.image_reward_utils import rm_load

    scorer = rm_load("ImageReward-v1.0")

    if not return_loss:
        def _fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer.score_batched(prompts, images)
            if bias:
                scores += bias
            return scores

        return _fn

    else:
        def loss_fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer.score_batched(prompts, images)

            loss = - scores
            return loss, scores

        return loss_fn


def PickScore(
    inference_dtype=None, 
    device=None, 
    return_loss=False, 
):
    from src.smc.scorers.PickScore_scorer import PickScoreScorer

    scorer = PickScoreScorer(dtype=torch.float32, device=device)
    scorer.requires_grad_(False)

    if not return_loss:
        def _fn(images, prompts):
            # from src.plot_utils import save_batch_images
            # save_batch_images(images, "output_SMC")
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)
            return scores

        return _fn

    else:
        def loss_fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)

            loss = - scores
            return loss, scores

        return loss_fn


def color_match_reward(x: torch.Tensor, target_color: torch.Tensor) -> torch.Tensor:
    """
    Reward images whose *mean* RGB comes close to a given target color.
    
    Args:
      x             : [B, 3, H, W] float images (e.g. in [0,1] or [0,255])
      target_color  : [3]    float tensor with your desired RGB mean
    
    Returns:
      reward        : [B]    higher when image mean-color ≈ target_color
    """
    B, C, H, W = x.shape
    # compute per-image mean color vector [B,3]
    mean_color = x.view(B, C, -1).mean(dim=2)
    
    # squared distance in RGB space
    dist2 = (mean_color - target_color[None, :].to(x.device)).pow(2).sum(dim=1)
    
    # negative distance = higher reward for closer color
    return -dist2
