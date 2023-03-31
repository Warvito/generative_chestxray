""" Script to compute the MS-SSIM score of the samples of the LDM.

In order to measure the diversity of the samples generated by the LDM, we use the Frechet Inception
Distance (FID) metric between 1000 images from the MIMIC-CXR dataset and 1000 images from the LDM.
"""
import argparse
from pathlib import Path

import torch
from generative.metrics import MultiScaleSSIMMetric
from monai import transforms
from monai.config import print_config
from monai.data import CacheDataset
from monai.utils import set_determinism
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2, help="Random seed to use.")
    parser.add_argument("--sample_dir", help="Location of the samples to evaluate.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")

    args = parser.parse_args()
    return args


def main(args):
    set_determinism(seed=args.seed)
    print_config()

    sample_dir = Path("/project/outputs/samples_unconditioned/")
    sample_list = sorted(list(sample_dir.glob("*.jpg")))

    datalist = []
    for sample_path in sample_list:
        datalist.append(
            {
                "image": str(sample_path),
            }
        )

    eval_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.EnsureChannelFirstd(keys=["image"]),
            transforms.Rotate90d(keys=["image"], k=-1, spatial_axes=(0, 1)),  # Fix flipped image read
            transforms.Flipd(keys=["image"], spatial_axis=1),  # Fix flipped image read
            transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
            transforms.ToTensord(keys=["image"]),
        ]
    )

    eval_ds = CacheDataset(
        data=datalist,
        transform=eval_transforms,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )

    eval_ds_2 = CacheDataset(
        data=datalist,
        transform=eval_transforms,
    )
    eval_loader_2 = DataLoader(
        eval_ds_2,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda")
    ms_ssim = MultiScaleSSIMMetric(spatial_dims=2, data_range=1.0)

    print("Computing MS-SSIM...")
    ms_ssim_list = []
    pbar = tqdm(enumerate(eval_loader), total=len(eval_loader))
    for step, batch in pbar:
        img = batch["image"]
        for batch2 in eval_loader_2:
            img2 = batch2["image"]
            if batch["image_meta_dict"]["filename_or_obj"][0] == batch2["image_meta_dict"]["filename_or_obj"][0]:
                continue
            ms_ssim_list.append(ms_ssim(img.to(device), img2.to(device)).item())
        pbar.update()

    ms_ssim_list = torch.cat(ms_ssim_list, axis=0)

    print(f"Mean MS-SSIM: {ms_ssim_list.mean():.6f}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
