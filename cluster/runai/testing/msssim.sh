seed=42
sample_dir="/project/outputs/samples_unconditioned/"
num_workers=8

runai submit \
  --name  mimic-ssim-sample \
  --image aicregistry:5000/wds20:ldm_mimic \
  --backoff-limit 0 \
  --gpu 1 \
  --cpu 4 \
  --large-shm \
  --run-as-user \
  --host-ipc \
  --project wds20 \
  --volume /nfs/home/wds20/projects/generative_mimic/:/project/ \
  --command -- python3 /project/src/python/testing/compute_msssim_sample.py \
      --seed=${seed} \
      --sample_dir=${sample_dir} \
      --num_workers=${num_workers}
