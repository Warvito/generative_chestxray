seed=2
run_dir="aekl_v3_ldm_v3"
training_ids="/project/outputs/ids/train.tsv"
validation_ids="/project/outputs/ids/validation.tsv"
stage1_uri="/project/mlruns/398344666374521908/d21dfcf5ba6d424dad18994bec47af29/artifacts/final_model"
config_file="/project/configs/ldm/ldm_v3.yaml"
scale_factor=0.3
batch_size=96
n_epochs=1000
eval_freq=10
num_workers=128
extended_report=0
experiment="LDM"

runai submit \
  --name  mimic-ldm-v3 \
  --image aicregistry:5000/wds20:ldm_mimic \
  --backoff-limit 0 \
  --gpu 8 \
  --cpu 64 \
  --large-shm \
  --run-as-user \
  --host-ipc \
  --project wds20 \
  --volume /nfs/home/wds20/projects/generative_mimic/:/project/ \
  --volume /nfs/home/wds20/datasets/MIMIC-CXR-JPG_v2.0.0/:/data/ \
  --command -- bash /project/src/bash/start_script.sh \
      python3 /project/src/python/training/train_ldm.py \
      seed=${seed} \
      run_dir=${run_dir} \
      training_ids=${training_ids} \
      validation_ids=${validation_ids} \
      stage1_uri=${stage1_uri} \
      config_file=${config_file} \
      scale_factor=${scale_factor} \
      batch_size=${batch_size} \
      n_epochs=${n_epochs} \
      eval_freq=${eval_freq} \
      num_workers=${num_workers} \
      extended_report=${extended_report} \
      experiment=${experiment}
