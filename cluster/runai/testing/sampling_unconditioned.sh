output_dir="/project/outputs/samples_unconditioned_v3/"
stage1_path="/project/outputs/models/v0.3/autoencoder.pth"
diffusion_path="/project/outputs/models/v0.3/diffusion_model.pth"
stage1_config_file_path="/project/configs/stage1/aekl_v3.yaml"
diffusion_config_file_path="/project/configs/ldm/ldm_v3.yaml"
prompt="uncondtioned"
guidance_scale=0.0
x_size=64
y_size=64
scale_factor=0.3
num_inference_steps=200

for i in {0..3}; do
  start_seed=$((i*250))
  stop_seed=$(((i+1)*250))
  runai submit \
    --name  sampling-mimic-${start_seed}-${stop_seed} \
    --image aicregistry:5000/wds20:ldm_mimic \
    --backoff-limit 0 \
    --gpu 1 \
    --cpu 4 \
    --memory-limit 256G \
    --node-type "A100" \
    --large-shm \
    --run-as-user \
    --host-ipc \
    --project wds20 \
    --volume /nfs/home/wds20/projects/generative_mimic/:/project/ \
    --command -- bash /project/src/bash/start_script.sh \
      python3 /project/src/python/testing/sample_images.py \
        output_dir=${output_dir} \
        stage1_path=${stage1_path} \
        diffusion_path=${diffusion_path} \
        stage1_config_file_path=${stage1_config_file_path} \
        diffusion_config_file_path=${diffusion_config_file_path} \
        start_seed=${start_seed} \
        stop_seed=${stop_seed} \
        prompt=${prompt} \
        guidance_scale=${guidance_scale} \
        x_size=${x_size} \
        y_size=${y_size} \
        scale_factor=${scale_factor} \
        num_inference_steps=${num_inference_steps}
done
