seed=42
run_dir="aekl_v0"
training_ids="/project/outputs/ids/train_ids.tsv"
validation_ids="/project/outputs/ids/val_ids.tsv"
config_file="/project/configs/stage1/aekl_v0.yaml"
batch_size=48
n_epochs=100
adv_start=5
eval_freq=3
num_workers=64
experiment="AEKL"

runai submit \
  --name mimic-aekl-v0 \
  --image aicregistry:5000/wds20:ldm_mimic \
  --backoff-limit 0 \
  --gpu 8 \
  --cpu 32 \
  --large-shm \
  --run-as-user \
  --node-type "A100" \
  --host-ipc \
  --project wds20 \
  --volume /nfs/home/wds20/projects/generative_mimic/:/project/ \
  --volume /nfs/home/wds20/datasets/MIMIC-CXR-JPG_v2.0.0/:/data/ \
  --command -- bash /project/src/bash/start_script.sh \
    python3 /project/src/python/training/train_aekl.py \
      seed=${seed} \
      run_dir=${run_dir} \
      training_ids=${training_ids} \
      validation_ids=${validation_ids} \
      config_file=${config_file} \
      batch_size=${batch_size} \
      n_epochs=${n_epochs} \
      adv_start=${adv_start} \
      eval_freq=${eval_freq} \
      num_workers=${num_workers} \
      experiment=${experiment}
