import matplotlib.pyplot as plt
import mlflow.pytorch
import torch
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDIMScheduler
from monai.config import print_config
from monai.utils import set_determinism
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

seed = 42
set_determinism(seed=seed)
print_config()

# output_dir = Path("/media/walter/Storage/Projects/generative_cardiac/outputs/figures/same_seed")
# output_dir.mkdir(exist_ok=True, parents=True)
#
stage1_old = mlflow.pytorch.load_model(
    "/media/walter/Storage/Projects/generative_mimic/mlruns/398344666374521908/6f280de5aa634aab96e6c31eed22a62b/artifacts/final_model"
)
stage1 = AutoencoderKL(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    num_channels=[64, 128, 128, 128],
    latent_channels=3,
    num_res_blocks=2,
    attention_levels=[False, False, False, False],
    with_encoder_nonlocal_attn=True,
    with_decoder_nonlocal_attn=True,
)
stage1.load_state_dict(stage1_old.state_dict())
stage1.eval()
del stage1_old

diffusion_old = mlflow.pytorch.load_model(
    "/media/walter/Storage/Projects/generative_mimic/mlruns/411881789846457862/6f1d5a773cf5421aadd7ff787bfe7643/artifacts/final_model"
)
diffusion = DiffusionModelUNet(
    spatial_dims=2,
    in_channels=3,
    out_channels=3,
    num_res_blocks=2,
    num_channels=[256, 512, 768],
    attention_levels=[False, True, True],
    with_conditioning=True,
    cross_attention_dim=1024,
    num_head_channels=[0, 512, 768],
)
diffusion.load_state_dict(diffusion_old.state_dict())
diffusion.eval()
del diffusion_old


device = torch.device("cuda")
diffusion = diffusion.to(device)
stage1 = stage1.to(device)

scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.0015,
    beta_end=0.0205,
    beta_schedule="scaled_linear",
    prediction_type="v_prediction",
    clip_sample=False,
)
scheduler.set_timesteps(200)

text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="text_encoder")
tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="tokenizer")

prompt = ["", "small right-sided pleural effusion"]
text_inputs = tokenizer(
    prompt,
    padding="max_length",
    max_length=tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt",
)
text_input_ids = text_inputs.input_ids

prompt_embeds = text_encoder(text_input_ids.squeeze(1))
prompt_embeds = prompt_embeds[0].to(device)

guidance_scale = 7.0
noise = torch.randn((1, 3, 64, 64)).to(device)

with torch.no_grad():
    progress_bar = tqdm(scheduler.timesteps)
    for t in progress_bar:
        noise_input = torch.cat([noise] * 2)
        model_output = diffusion(
            noise_input, timesteps=torch.Tensor((t,)).to(noise.device).long(), context=prompt_embeds
        )
        noise_pred_uncond, noise_pred_text = model_output.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        noise, _ = scheduler.step(noise_pred, t, noise)

with torch.no_grad():
    sample = stage1.decode_stage_2_outputs(noise / 0.3)


plt.imshow(sample.cpu()[0, 0, :, :], cmap="gray", vmin=0, vmax=1)
plt.show()


torch.save(
    diffusion.state_dict(),
    "/media/walter/Storage/Projects/GenerativeModels/model-zoo/models/cxr_image_synthesis_latent_diffusion_model/models/diffusion_model.pth",
)
torch.save(
    stage1.state_dict(),
    "/media/walter/Storage/Projects/GenerativeModels/model-zoo/models/cxr_image_synthesis_latent_diffusion_model/models/autoencoder.pth",
)
