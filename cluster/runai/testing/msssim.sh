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
  --command -- python3 /project/src/python/testing/compute_msssim_test_set.py \
      --seed=${seed} \
      --test_ids=${test_ids} \
      --num_workers=${num_workers}
