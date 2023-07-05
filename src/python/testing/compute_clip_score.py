""" Compute CLIP score using BiomedCLIP.

Based on https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
"""

import argparse
from pathlib import Path

import open_clip
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from monai.config import print_config
from PIL import Image
from tqdm import trange


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--samples_dir", help="Path to save the .pth file of the diffusion model.")
    parser.add_argument("--output_dir", help="Path to the MLFlow artifact of the stage1.")

    args = parser.parse_args()
    return args


def main(args):
    print_config()

    samples_dir = Path(args.samples_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load BiomedCLIP model
    model, _, preprocess_val = open_clip.create_model_and_transforms(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    tokenizer = open_clip.get_tokenizer("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    snapshot_download("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", local_dir="biomed-clip-share")
    context_length = 256

    device = torch.device("cuda")
    model.to(device)
    model.eval()

    clip_scores = []
    for i in trange(1000):
        img = Image.open(samples_dir / f"sample_{i}.png")
        images = torch.stack([preprocess_val(img)]).to(device)

        prompt = ""
        if i < 125:
            prompt = "atelectasis chest x-ray"
        elif i >= 125 and i < 250:
            prompt = "cardiomegaly chest x-ray"
        elif i >= 250 and i < 375:
            prompt = "normal chest x-ray"
        elif i >= 375 and i < 500:
            prompt = "edema chest x-ray"
        elif i >= 500 and i < 625:
            prompt = "enlarged cardiomediastinum chest x-ray"
        elif i >= 625 and i < 750:
            prompt = "pleural effusion chest x-ray"
        elif i >= 750 and i < 875:
            prompt = "pneumonia chest x-ray"
        elif i >= 875:
            prompt = "pneumothorax chest x-ray"

        texts = tokenizer([prompt], context_length=context_length).to(device)
        with torch.no_grad():
            image_features, text_features, logit_scale = model(images, texts)
            clip_score = (image_features * text_features).sum(axis=-1)
            clip_scores.append(clip_score.item())

    prediction_df = pd.DataFrame({"clip_score": clip_scores})
    prediction_df.to_csv(output_dir / "clip_scores.tsv", index=False, sep="\t")


if __name__ == "__main__":
    args = parse_args()
    main(args)
