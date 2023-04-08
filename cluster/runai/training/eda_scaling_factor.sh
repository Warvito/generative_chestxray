seed=2
training_ids="/project/outputs/ids/train.tsv"
validation_ids="/project/outputs/ids/validation.tsv"
stage1_uri="/project/mlruns/398344666374521908/0fde76e3e71b4ed4a92aea593c73c3db/artifacts/final_model"
batch_size=32
num_workers=8

runai submit \
  --name  eda-scaling-factor \
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
      python3 /project/src/python/training/eda_ldm_scaling_factor.py \
      seed=${seed} \
      training_ids=${training_ids} \
      validation_ids=${validation_ids} \
      stage1_uri=${stage1_uri} \
      batch_size=${batch_size} \
      num_workers=${num_workers}
