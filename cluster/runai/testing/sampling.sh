output_dir="/project/outputs/samples/"
stage1_path="/project/outputs/models/v0.2/autoencoder.pth"
diffusion_path="/project/outputs/models/v0.2/diffusion_model.pth"
stage1_config_file_path="/project/configs/stage1/aekl_v0.yaml"
diffusion_config_file_path="/project/configs/ldm/ldm_v0.yaml"
start_seed=0
stop_seed=100
prompt="atelectasis"
guidance_scale=7.0
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

output_dir="/project/outputs/samples/"
stage1_path="/project/outputs/models/v0.2/autoencoder.pth"
diffusion_path="/project/outputs/models/v0.2/diffusion_model.pth"
stage1_config_file_path="/project/configs/stage1/aekl_v0.yaml"
diffusion_config_file_path="/project/configs/ldm/ldm_v0.yaml"
start_seed=100
stop_seed=200
prompt="cardiomegaly"
guidance_scale=7.0
x_size=64
y_size=64
scale_factor=0.3
num_inference_steps=200

runai submit \
  --name  sampling-mimic-1 \
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


output_dir="/project/outputs/samples/"
stage1_path="/project/outputs/models/v0.2/autoencoder.pth"
diffusion_path="/project/outputs/models/v0.2/diffusion_model.pth"
stage1_config_file_path="/project/configs/stage1/aekl_v0.yaml"
diffusion_config_file_path="/project/configs/ldm/ldm_v0.yaml"
start_seed=200
stop_seed=300
prompt="consolidation"
guidance_scale=7.0
x_size=64
y_size=64
scale_factor=0.3
num_inference_steps=200

runai submit \
  --name  sampling-mimic-2 \
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


output_dir="/project/outputs/samples/"
stage1_path="/project/outputs/models/v0.2/autoencoder.pth"
diffusion_path="/project/outputs/models/v0.2/diffusion_model.pth"
stage1_config_file_path="/project/configs/stage1/aekl_v0.yaml"
diffusion_config_file_path="/project/configs/ldm/ldm_v0.yaml"
start_seed=300
stop_seed=400
prompt="edema"
guidance_scale=7.0
x_size=64
y_size=64
scale_factor=0.3
num_inference_steps=200

runai submit \
  --name  sampling-mimic-3 \
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


output_dir="/project/outputs/samples/"
stage1_path="/project/outputs/models/v0.2/autoencoder.pth"
diffusion_path="/project/outputs/models/v0.2/diffusion_model.pth"
stage1_config_file_path="/project/configs/stage1/aekl_v0.yaml"
diffusion_config_file_path="/project/configs/ldm/ldm_v0.yaml"
start_seed=400
stop_seed=500
prompt="no_findings"
guidance_scale=7.0
x_size=64
y_size=64
scale_factor=0.3
num_inference_steps=200

runai submit \
  --name  sampling-mimic-4 \
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



output_dir="/project/outputs/samples/"
stage1_path="/project/outputs/models/v0.2/autoencoder.pth"
diffusion_path="/project/outputs/models/v0.2/diffusion_model.pth"
stage1_config_file_path="/project/configs/stage1/aekl_v0.yaml"
diffusion_config_file_path="/project/configs/ldm/ldm_v0.yaml"
start_seed=500
stop_seed=600
prompt="enlarged_cardiomediastinum"
guidance_scale=7.0
x_size=64
y_size=64
scale_factor=0.3
num_inference_steps=200

runai submit \
  --name  sampling-mimic-5 \
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


output_dir="/project/outputs/samples/"
stage1_path="/project/outputs/models/v0.2/autoencoder.pth"
diffusion_path="/project/outputs/models/v0.2/diffusion_model.pth"
stage1_config_file_path="/project/configs/stage1/aekl_v0.yaml"
diffusion_config_file_path="/project/configs/ldm/ldm_v0.yaml"
start_seed=600
stop_seed=700
prompt="fracture"
guidance_scale=7.0
x_size=64
y_size=64
scale_factor=0.3
num_inference_steps=200

runai submit \
  --name  sampling-mimic-6 \
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


output_dir="/project/outputs/samples/"
stage1_path="/project/outputs/models/v0.2/autoencoder.pth"
diffusion_path="/project/outputs/models/v0.2/diffusion_model.pth"
stage1_config_file_path="/project/configs/stage1/aekl_v0.yaml"
diffusion_config_file_path="/project/configs/ldm/ldm_v0.yaml"
start_seed=700
stop_seed=800
prompt="lung_lesion"
guidance_scale=7.0
x_size=64
y_size=64
scale_factor=0.3
num_inference_steps=200

runai submit \
  --name  sampling-mimic-7 \
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


output_dir="/project/outputs/samples/"
stage1_path="/project/outputs/models/v0.2/autoencoder.pth"
diffusion_path="/project/outputs/models/v0.2/diffusion_model.pth"
stage1_config_file_path="/project/configs/stage1/aekl_v0.yaml"
diffusion_config_file_path="/project/configs/ldm/ldm_v0.yaml"
start_seed=800
stop_seed=900
prompt="lung_opacity"
guidance_scale=7.0
x_size=64
y_size=64
scale_factor=0.3
num_inference_steps=200

runai submit \
  --name  sampling-mimic-8 \
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


output_dir="/project/outputs/samples/"
stage1_path="/project/outputs/models/v0.2/autoencoder.pth"
diffusion_path="/project/outputs/models/v0.2/diffusion_model.pth"
stage1_config_file_path="/project/configs/stage1/aekl_v0.yaml"
diffusion_config_file_path="/project/configs/ldm/ldm_v0.yaml"
start_seed=800
stop_seed=900
prompt="pleural_effusion"
guidance_scale=7.0
x_size=64
y_size=64
scale_factor=0.3
num_inference_steps=200

runai submit \
  --name  sampling-mimic-9 \
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



output_dir="/project/outputs/samples/"
stage1_path="/project/outputs/models/v0.2/autoencoder.pth"
diffusion_path="/project/outputs/models/v0.2/diffusion_model.pth"
stage1_config_file_path="/project/configs/stage1/aekl_v0.yaml"
diffusion_config_file_path="/project/configs/ldm/ldm_v0.yaml"
start_seed=900
stop_seed=1000
prompt="pneumonia"
guidance_scale=7.0
x_size=64
y_size=64
scale_factor=0.3
num_inference_steps=200

runai submit \
  --name  sampling-mimic-12 \
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


output_dir="/project/outputs/samples/"
stage1_path="/project/outputs/models/v0.2/autoencoder.pth"
diffusion_path="/project/outputs/models/v0.2/diffusion_model.pth"
stage1_config_file_path="/project/configs/stage1/aekl_v0.yaml"
diffusion_config_file_path="/project/configs/ldm/ldm_v0.yaml"
start_seed=1000
stop_seed=1100
prompt="pneumothorax"
guidance_scale=7.0
x_size=64
y_size=64
scale_factor=0.3
num_inference_steps=200

runai submit \
  --name  sampling-mimic-10 \
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


output_dir="/project/outputs/samples/"
stage1_path="/project/outputs/models/v0.2/autoencoder.pth"
diffusion_path="/project/outputs/models/v0.2/diffusion_model.pth"
stage1_config_file_path="/project/configs/stage1/aekl_v0.yaml"
diffusion_config_file_path="/project/configs/ldm/ldm_v0.yaml"
start_seed=1100
stop_seed=1200
prompt="support_devices"
guidance_scale=7.0
x_size=64
y_size=64
scale_factor=0.3
num_inference_steps=200

runai submit \
  --name  sampling-mimic-11 \
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
