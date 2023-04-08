""" Script to perform an exploratory data analysis to find the appropriate scaling factor for the diffusion model. """
import argparse
import warnings
from pathlib import Path

import mlflow.pytorch
import torch
from monai.config import print_config
from monai.utils import first, set_determinism
from util import get_dataloader

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2, help="Random seed to use.")
    parser.add_argument("--training_ids", help="Location of file with training ids.")
    parser.add_argument("--validation_ids", help="Location of file with validation ids.")
    parser.add_argument("--stage1_uri", help="Path readable by load_model.")
    parser.add_argument("--batch_size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")

    args = parser.parse_args()
    return args


def main(args):
    set_determinism(seed=args.seed)
    print_config()

    print(f"Arguments: {str(args)}")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    output_dir = Path("/project/outputs/runs/")
    output_dir.mkdir(exist_ok=True, parents=True)

    print("Getting data...")
    cache_dir = output_dir / "cached_data_eda"
    cache_dir.mkdir(exist_ok=True)

    train_loader, _ = get_dataloader(
        cache_dir=cache_dir,
        batch_size=args.batch_size,
        training_ids=args.training_ids,
        validation_ids=args.validation_ids,
        num_workers=args.num_workers,
        model_type="diffusion",
    )

    # Load Autoencoder to produce the latent representations
    print(f"Loading Stage 1 from {args.stage1_uri}")
    device = torch.device("cuda")
    stage1 = mlflow.pytorch.load_model(args.stage1_uri)
    stage1.eval()
    stage1 = stage1.to(device)

    eda_data = first(train_loader)["image"]

    with torch.no_grad():
        z = stage1.encode_stage_2_inputs(eda_data.to(device))

    print(f"Scaling factor: {1 / torch.std(z)}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
