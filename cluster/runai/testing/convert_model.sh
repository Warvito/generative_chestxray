stage1_mlflow_path="/project/mlruns/398344666374521908/d21dfcf5ba6d424dad18994bec47af29/artifacts/final_model"
diffusion_mlflow_path="/project/mlruns/411881789846457862/c8264f93832b41f5bd94f0c32adbb2e2/artifacts/final_model"
output_dir="/project/outputs/models/v0.3"

runai submit \
  --name mimic-convert-model \
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
    python3 /project/src/python/testing/convert_mlflow_to_pytorch.py \
      stage1_mlflow_path=${stage1_mlflow_path} \
      diffusion_mlflow_path=${diffusion_mlflow_path} \
      output_dir=${output_dir}
