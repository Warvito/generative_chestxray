seed=2
run_dir="aekl_v1_ldm_v0"
training_ids="/project/outputs/ids/training.tsv"
validation_ids="/project/outputs/ids/validation.tsv"
stage1_uri="/project/mlruns/513041168923337346/267bedf8592c42419854f94ca3d77c27/artifacts/final_model"
config_file="/project/configs/ldm/ldm_v0.yaml"
batch_size=48
n_epochs=300
eval_freq=5
num_workers=16
experiment="LDM"

runai submit \
  --name  ldm-boneage-v0 \
  --image aicregistry:5000/wds20:ldms2 \
  --backoff-limit 0 \
  --gpu 2 \
  --cpu 8 \
  --large-shm \
  --run-as-user \
  --host-ipc \
  --project wds20 \
  --volume /nfs/home/wds20/projects/generative_cardiac/:/project/ \
  --volume /nfs/home/wds20/datasets/CARDIAC_UKBIOBANK/rawdata:/data/ \
  --command -- bash /project/src/bash/start_script.sh \
      python3 /project/src/python/training/train_ldm.py \
      seed=${seed} \
      run_dir=${run_dir} \
      training_ids=${training_ids} \
      validation_ids=${validation_ids} \
      stage1_uri=${stage1_uri} \
      config_file=${config_file} \
      batch_size=${batch_size} \
      n_epochs=${n_epochs} \
      eval_freq=${eval_freq} \
      num_workers=${num_workers} \
      experiment=${experiment}
