output_dir="/project/outputs/metrics/v0.3"
test_ids="/project/outputs/ids/test.tsv"
config_file="/project/configs/stage1/aekl_v3.yaml"
stage1_path="/project/outputs/models/v0.3/autoencoder.pth"
seed=42
batch_size=16
num_workers=8

runai submit \
  --name  mimic-ssim \
  --image aicregistry:5000/wds20:ldm_mimic \
  --backoff-limit 0 \
  --gpu 1 \
  --cpu 4 \
  --large-shm \
  --run-as-user \
  --host-ipc \
  --project wds20 \
  --volume /nfs/home/wds20/projects/generative_mimic/:/project/ \
  --volume /nfs/home/wds20/datasets/MIMIC-CXR-JPG_v2.0.0/:/data/ \
  --command -- bash /project/src/bash/start_script.sh \
    python3 /project/src/python/testing/compute_msssim_reconstruction.py \
      seed=${seed} \
      output_dir=${output_dir} \
      test_ids=${test_ids} \
      batch_size=${batch_size} \
      config_file=${config_file} \
      stage1_path=${stage1_path} \
      num_workers=${num_workers}
