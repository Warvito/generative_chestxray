from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from pynvml.smi import nvidia_smi
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from util import log_reconstructions


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def print_gpu_memory_report():
    if torch.cuda.is_available():
        nvsmi = nvidia_smi.getInstance()
        data = nvsmi.DeviceQuery("memory.used, memory.total, utilization.gpu")["gpu"]
        print("Memory report")
        for i, data_by_rank in enumerate(data):
            mem_report = data_by_rank["fb_memory_usage"]
            print(f"gpu:{i} mem(%) {int(mem_report['used'] * 100.0 / mem_report['total'])}")


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) + torch.mean(torch.nn.functional.softplus(logits_fake))
    )
    return d_loss


def calculate_adaptive_weight(nll_loss, g_loss, discriminator_weight, last_layer):
    nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
    g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
    d_weight = d_weight * discriminator_weight
    return d_weight


# ----------------------------------------------------------------------------------------------------------------------
# AUTOENCODER KL
# ----------------------------------------------------------------------------------------------------------------------
def train_aekl(
    model: nn.Module,
    discriminator: nn.Module,
    perceptual_loss: nn.Module,
    start_epoch: int,
    best_loss: float,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    n_epochs: int,
    epoch_disc_start: int,
    eval_freq: int,
    writer_train: SummaryWriter,
    writer_val: SummaryWriter,
    device: torch.device,
    run_dir: Path,
    adv_weight: float,
    perceptual_weight: float,
    kl_weight: float,
) -> float:
    scaler_g = GradScaler()
    scaler_d = GradScaler()

    raw_model = model.module if hasattr(model, "module") else model

    val_loss = eval_aekl(
        model=model,
        discriminator=discriminator,
        perceptual_loss=perceptual_loss,
        loader=val_loader,
        device=device,
        step=len(train_loader) * start_epoch,
        step_disc_start=len(train_loader) * epoch_disc_start,
        writer=writer_val,
        kl_weight=kl_weight,
        adv_weight=adv_weight,
        perceptual_weight=perceptual_weight,
    )
    print(f"epoch {start_epoch} val loss: {val_loss:.4f}")
    for epoch in range(start_epoch, n_epochs):
        train_epoch_aekl(
            model=model,
            discriminator=discriminator,
            perceptual_loss=perceptual_loss,
            loader=train_loader,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            device=device,
            epoch=epoch,
            epoch_disc_start=epoch_disc_start,
            writer=writer_train,
            kl_weight=kl_weight,
            adv_weight=adv_weight,
            perceptual_weight=perceptual_weight,
            scaler_g=scaler_g,
            scaler_d=scaler_d,
        )

        if (epoch + 1) % eval_freq == 0:
            val_loss = eval_aekl(
                model=model,
                discriminator=discriminator,
                perceptual_loss=perceptual_loss,
                loader=val_loader,
                device=device,
                step=len(train_loader) * epoch,
                step_disc_start=len(train_loader) * epoch_disc_start,
                writer=writer_val,
                kl_weight=kl_weight,
                adv_weight=adv_weight,
                perceptual_weight=perceptual_weight,
            )
            print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")
            print_gpu_memory_report()

            # Save checkpoint
            checkpoint = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "discriminator": discriminator.state_dict(),
                "optimizer_g": optimizer_g.state_dict(),
                "optimizer_d": optimizer_d.state_dict(),
                "best_loss": best_loss,
            }
            torch.save(checkpoint, str(run_dir / "checkpoint.pth"))

            if val_loss <= best_loss:
                print(f"New best val loss {val_loss}")
                best_loss = val_loss

    print(f"Training finished!")
    print(f"Saving final model...")
    torch.save(raw_model.state_dict(), str(run_dir / "final_model.pth"))

    return val_loss


def train_epoch_aekl(
    model: nn.Module,
    discriminator: nn.Module,
    perceptual_loss: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    epoch_disc_start: int,
    writer: SummaryWriter,
    kl_weight: float,
    adv_weight: float,
    perceptual_weight: float,
    scaler_g: GradScaler,
    scaler_d: GradScaler,
) -> None:

    model.train()
    discriminator.train()

    pbar = tqdm(enumerate(loader), total=len(loader))
    for step, x in pbar:
        images = x["image"].to(device)

        # GENERATOR
        optimizer_g.zero_grad(set_to_none=True)
        with autocast(enabled=True):
            reconstruction, z_mu, z_sigma = model(x=images)
            l1_loss = F.l1_loss(reconstruction.float(), images.float())
            p_loss = perceptual_loss(reconstruction.float(), images.float())

            kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

            logits_fake = discriminator(reconstruction.contiguous().float())
            generator_loss = -torch.mean(logits_fake)

            rec_loss = l1_loss + perceptual_weight * p_loss
            rec_loss = rec_loss.mean()
            #
            # try:
            #     d_weight = calculate_adaptive_weight(rec_loss, generator_loss, last_layer=model.decoder.blocks[-1].conv.weight[0])
            # except RuntimeError:
            #     assert not model.training
            #     d_weight = torch.tensor(0.0)

            loss = l1_loss + kl_weight * kl_loss + perceptual_weight * p_loss + adv_weight * generator_loss

            loss = loss.mean()
            l1_loss = l1_loss.mean()
            p_loss = p_loss.mean()
            kl_loss = kl_loss.mean()
            g_loss = generator_loss.mean()

            losses = OrderedDict(
                loss=loss,
                l1_loss=l1_loss,
                p_loss=p_loss,
                kl_loss=kl_loss,
                g_loss=g_loss,
            )

        scaler_g.scale(losses["loss"]).backward()
        scaler_g.unscale_(optimizer_g)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        scaler_g.step(optimizer_g)
        scaler_g.update()

        # DISCRIMINATOR
        optimizer_d.zero_grad(set_to_none=True)

        with autocast(enabled=True):
            logits_fake = discriminator(reconstruction.contiguous().detach())
            fake_label = torch.zeros_like(logits_fake, device=logits_fake.device)
            loss_d_fake = F.mse_loss(logits_fake, fake_label)
            logits_real = discriminator(images.contiguous().detach())
            real_label = torch.ones_like(logits_real, device=logits_real.device)
            loss_d_real = F.mse_loss(logits_real, real_label)
            discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

            d_loss = adv_weight * discriminator_loss
            d_loss = d_loss.mean()

        scaler_d.scale(d_loss).backward()
        scaler_d.unscale_(optimizer_d)
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1)
        scaler_d.step(optimizer_d)
        scaler_d.update()

        losses["d_loss"] = discriminator_loss

        writer.add_scalar("lr_g", get_lr(optimizer_g), epoch * len(loader) + step)
        writer.add_scalar("lr_d", get_lr(optimizer_d), epoch * len(loader) + step)
        for k, v in losses.items():
            writer.add_scalar(f"{k}", v.item(), epoch * len(loader) + step)

        pbar.set_postfix(
            {
                "epoch": epoch,
                "loss": f"{losses['loss'].item():.6f}",
                "l1_loss": f"{losses['l1_loss'].item():.6f}",
                "p_loss": f"{losses['p_loss'].item():.6f}",
                "g_loss": f"{losses['g_loss'].item():.6f}",
                "d_loss": f"{losses['d_loss'].item():.6f}",
                "lr_g": f"{get_lr(optimizer_g):.6f}",
                "lr_d": f"{get_lr(optimizer_d):.6f}",
            },
        )


@torch.no_grad()
def eval_aekl(
    model: nn.Module,
    discriminator: nn.Module,
    perceptual_loss: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    step: int,
    writer: SummaryWriter,
    kl_weight: float,
    adv_weight: float,
    perceptual_weight: float,
) -> float:
    model.eval()
    discriminator.eval()

    total_losses = OrderedDict()
    for x in loader:
        images = x["image"].to(device)

        with autocast(enabled=True):
            # GENERATOR
            reconstruction, z_mu, z_sigma = model(x=images)
            l1_loss = F.l1_loss(reconstruction.float(), images.float())
            p_loss = perceptual_loss(reconstruction.float(), images.float())
            kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

            logits_fake = discriminator(reconstruction.contiguous().float())
            real_label = torch.ones_like(logits_fake, device=logits_fake.device)
            generator_loss = F.mse_loss(logits_fake, real_label)

            # DISCRIMINATOR
            logits_fake = discriminator(reconstruction.contiguous().detach())
            fake_label = torch.zeros_like(logits_fake, device=logits_fake.device)
            loss_d_fake = F.mse_loss(logits_fake, fake_label)
            logits_real = discriminator(images.contiguous().detach())
            real_label = torch.ones_like(logits_real, device=logits_real.device)
            loss_d_real = F.mse_loss(logits_real, real_label)
            discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

            loss = l1_loss + kl_weight * kl_loss + perceptual_weight * p_loss + adv_weight * generator_loss

            loss = loss.mean()
            l1_loss = l1_loss.mean()
            p_loss = p_loss.mean()
            kl_loss = kl_loss.mean()
            g_loss = generator_loss.mean()
            d_loss = discriminator_loss.mean()

            losses = OrderedDict(
                loss=loss,
                l1_loss=l1_loss,
                p_loss=p_loss,
                kl_loss=kl_loss,
                g_loss=g_loss,
                d_loss=d_loss,
            )

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0) + v.item() * images.shape[0]

    for k in total_losses.keys():
        total_losses[k] /= len(loader.dataset)

    for k, v in total_losses.items():
        writer.add_scalar(f"{k}", v, step)

    log_reconstructions(
        image=images,
        reconstruction=reconstruction,
        writer=writer,
        step=step,
    )

    return total_losses["l1_loss"]
