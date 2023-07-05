stage1_path="/project/outputs/models/v0.3/autoencoder.pth"
diffusion_path="/project/outputs/models/v0.3/diffusion_model.pth"
stage1_config_file_path="/project/configs/stage1/aekl_v3.yaml"
diffusion_config_file_path="/project/configs/ldm/ldm_v3.yaml"
start_seed=0
stop_seed=125
prompt="atelectasis"
x_size=64
y_size=64
scale_factor=0.3
num_inference_steps=200

counter=0
for guidance_scale in 1.0 1.5 1.75 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 ; do
    output_dir="/project/outputs/samples_fid_v03/guidance_scale_${guidance_scale}"
    runai submit \
      --name  mimic-sampling-0-${counter} \
      --image aicregistry:5000/wds20:ldm_mimic \
      --backoff-limit 0 \
      --gpu 1 \
      --cpu 4 \
      --memory-limit 256G \
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
    counter=$((counter+1))
    sleep 3
done

stage1_path="/project/outputs/models/v0.3/autoencoder.pth"
diffusion_path="/project/outputs/models/v0.3/diffusion_model.pth"
stage1_config_file_path="/project/configs/stage1/aekl_v3.yaml"
diffusion_config_file_path="/project/configs/ldm/ldm_v3.yaml"
start_seed=125
stop_seed=250
prompt="cardiomegaly"
x_size=64
y_size=64
scale_factor=0.3
num_inference_steps=200

counter=0
for guidance_scale in 1.0 1.5 1.75 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 ; do
    output_dir="/project/outputs/samples_fid_v03/guidance_scale_${guidance_scale}"
    runai submit \
      --name  mimic-sampling-1-${counter} \
      --image aicregistry:5000/wds20:ldm_mimic \
      --backoff-limit 0 \
      --gpu 1 \
      --cpu 4 \
      --memory-limit 256G \
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
    counter=$((counter+1))
    sleep 3
done


stage1_path="/project/outputs/models/v0.3/autoencoder.pth"
diffusion_path="/project/outputs/models/v0.3/diffusion_model.pth"
stage1_config_file_path="/project/configs/stage1/aekl_v3.yaml"
diffusion_config_file_path="/project/configs/ldm/ldm_v3.yaml"
start_seed=250
stop_seed=375
prompt="no_findings"
x_size=64
y_size=64
scale_factor=0.3
num_inference_steps=200

counter=0
for guidance_scale in 1.0 1.5 1.75 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 ; do
    output_dir="/project/outputs/samples_fid_v03/guidance_scale_${guidance_scale}"
    runai submit \
      --name  mimic-sampling-2-${counter} \
      --image aicregistry:5000/wds20:ldm_mimic \
      --backoff-limit 0 \
      --gpu 1 \
      --cpu 4 \
      --memory-limit 256G \
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
    counter=$((counter+1))
    sleep 3
done

stage1_path="/project/outputs/models/v0.3/autoencoder.pth"
diffusion_path="/project/outputs/models/v0.3/diffusion_model.pth"
stage1_config_file_path="/project/configs/stage1/aekl_v3.yaml"
diffusion_config_file_path="/project/configs/ldm/ldm_v3.yaml"
start_seed=375
stop_seed=500
prompt="edema"
x_size=64
y_size=64
scale_factor=0.3
num_inference_steps=200

counter=0
for guidance_scale in 1.0 1.5 1.75 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 ; do
    output_dir="/project/outputs/samples_fid_v03/guidance_scale_${guidance_scale}"
    runai submit \
      --name  mimic-sampling-3-${counter} \
      --image aicregistry:5000/wds20:ldm_mimic \
      --backoff-limit 0 \
      --gpu 1 \
      --cpu 4 \
      --memory-limit 256G \
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
    counter=$((counter+1))
    sleep 3
done

stage1_path="/project/outputs/models/v0.3/autoencoder.pth"
diffusion_path="/project/outputs/models/v0.3/diffusion_model.pth"
stage1_config_file_path="/project/configs/stage1/aekl_v3.yaml"
diffusion_config_file_path="/project/configs/ldm/ldm_v3.yaml"
start_seed=500
stop_seed=625
prompt="enlarged_cardiomediastinum"
x_size=64
y_size=64
scale_factor=0.3
num_inference_steps=200

counter=0
for guidance_scale in 1.0 1.5 1.75 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 ; do
    output_dir="/project/outputs/samples_fid_v03/guidance_scale_${guidance_scale}"
    runai submit \
      --name  mimic-sampling-4-${counter} \
      --image aicregistry:5000/wds20:ldm_mimic \
      --backoff-limit 0 \
      --gpu 1 \
      --cpu 4 \
      --memory-limit 256G \
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
    counter=$((counter+1))
    sleep 3
done

stage1_path="/project/outputs/models/v0.3/autoencoder.pth"
diffusion_path="/project/outputs/models/v0.3/diffusion_model.pth"
stage1_config_file_path="/project/configs/stage1/aekl_v3.yaml"
diffusion_config_file_path="/project/configs/ldm/ldm_v3.yaml"
start_seed=625
stop_seed=750
prompt="pleural_effusion"
x_size=64
y_size=64
scale_factor=0.3
num_inference_steps=200

counter=0
for guidance_scale in 1.0 1.5 1.75 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 ; do
    output_dir="/project/outputs/samples_fid_v03/guidance_scale_${guidance_scale}"
    runai submit \
      --name  mimic-sampling-5-${counter} \
      --image aicregistry:5000/wds20:ldm_mimic \
      --backoff-limit 0 \
      --gpu 1 \
      --cpu 4 \
      --memory-limit 256G \
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
    counter=$((counter+1))
    sleep 3
done

stage1_path="/project/outputs/models/v0.3/autoencoder.pth"
diffusion_path="/project/outputs/models/v0.3/diffusion_model.pth"
stage1_config_file_path="/project/configs/stage1/aekl_v3.yaml"
diffusion_config_file_path="/project/configs/ldm/ldm_v3.yaml"
start_seed=750
stop_seed=875
prompt="pneumonia"
x_size=64
y_size=64
scale_factor=0.3
num_inference_steps=200

counter=0
for guidance_scale in 1.0 1.5 1.75 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 ; do
    output_dir="/project/outputs/samples_fid_v03/guidance_scale_${guidance_scale}"
    runai submit \
      --name  mimic-sampling-6-${counter} \
      --image aicregistry:5000/wds20:ldm_mimic \
      --backoff-limit 0 \
      --gpu 1 \
      --cpu 4 \
      --memory-limit 256G \
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
    counter=$((counter+1))
    sleep 3
done


stage1_path="/project/outputs/models/v0.3/autoencoder.pth"
diffusion_path="/project/outputs/models/v0.3/diffusion_model.pth"
stage1_config_file_path="/project/configs/stage1/aekl_v3.yaml"
diffusion_config_file_path="/project/configs/ldm/ldm_v3.yaml"
start_seed=875
stop_seed=1000
prompt="pneumothorax"
x_size=64
y_size=64
scale_factor=0.3
num_inference_steps=200

counter=0
for guidance_scale in 1.0 1.5 1.75 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 ; do
    output_dir="/project/outputs/samples_fid_v03/guidance_scale_${guidance_scale}"
    runai submit \
      --name  mimic-sampling-7-${counter} \
      --image aicregistry:5000/wds20:ldm_mimic \
      --backoff-limit 0 \
      --gpu 1 \
      --cpu 4 \
      --memory-limit 256G \
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
    counter=$((counter+1))
    sleep 3
done
