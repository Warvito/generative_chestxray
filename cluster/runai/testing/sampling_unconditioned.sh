output_dir="/project/outputs/samples_unconditioned/"
stage1_path="/project/outputs/models/v0.2/autoencoder.pth"
diffusion_path="/project/outputs/models/v0.2/diffusion_model.pth"
stage1_config_file_path="/project/configs/stage1/aekl_v0.yaml"
diffusion_config_file_path="/project/configs/ldm/ldm_v0.yaml"
start_seed=0
stop_seed=1000
prompt="uncondtioned"
guidance_scale=0.0
x_size=64
y_size=64
scale_factor=0.3
num_inference_steps=200

runai submit \
  --name  sampling-mimic-0 \
  --image aicregistry:5000/wds20:ldm_mimic \
  --backoff-limit 0 \
  --gpu 1 \
  --cpu 4 \
  --large-shm \
  --run-as-user \
  --host-ipc \
  --project wds20 \
  --volume /nfs/home/wds20/projects/generative_mimic/:/project/ \
  --command -- python3 /project/src/python/testing/sample_images.py \
      --output_dir=${output_dir} \
      --stage1_path=${stage1_path} \
      --diffusion_path=${diffusion_path} \
      --stage1_config_file_path=${stage1_config_file_path} \
      --diffusion_config_file_path=${diffusion_config_file_path} \
      --start_seed=${start_seed} \
      --stop_seed=${stop_seed} \
      --prompt=${prompt} \
      --guidance_scale=${guidance_scale} \
      --x_size=${x_size} \
      --y_size=${y_size} \
      --scale_factor=${scale_factor} \
      --num_inference_steps=${num_inference_steps}
