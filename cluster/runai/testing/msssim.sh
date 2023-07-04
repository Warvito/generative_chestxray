seed=42
sample_dir="/project/outputs/samples_unconditioned_v3/"
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
  --command -- bash /project/src/bash/start_script.sh \
    python3 /project/src/python/testing/compute_msssim_sample.py \
      seed=${seed} \
      sample_dir=${sample_dir} \
      num_workers=${num_workers}

seed=42
test_ids="/project/outputs/ids/test.tsv"
num_workers=8

runai submit \
  --name  mimic-ssim-test-set \
  --image aicregistry:5000/wds20:ldm_mimic \
  --backoff-limit 0 \
  --gpu 1 \
  --cpu 8 \
  --large-shm \
  --run-as-user \
  --host-ipc \
  --project wds20 \
  --volume /nfs/home/wds20/projects/generative_mimic/:/project/ \
  --volume /nfs/home/wds20/datasets/MIMIC-CXR-JPG_v2.0.0/:/data/ \
  --command -- bash /project/src/bash/start_script.sh \
    python3 /project/src/python/testing/compute_msssim_test_set.py \
      seed=${seed} \
      test_ids=${test_ids} \
      num_workers=${num_workers}
