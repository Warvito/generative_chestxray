seed=42
run_dir="aekl_v0"
training_ids="/project/outputs/ids/training.tsv"
validation_ids="/project/outputs/ids/validation.tsv"
config_file="/project/configs/stage1/aekl_v0.yaml"
batch_size=4
n_epochs=500
eval_freq=10
num_workers=32
use_bfloat=0
experiment="AEKL"

runai submit \
  --name s2-aekl-v0 \
  --image aicregistry:5000/wds20:ldms2 \
  --backoff-limit 0 \
  --gpu 4 \
  --cpu 16 \
  --large-shm \
  --run-as-user \
  --node-type "A100" \
  --host-ipc \
  --project wds20 \
  --volume /nfs/home/wds20/projects/generative_cardiac/:/project/ \
  --volume /nfs/home/wds20/datasets/CARDIAC_UKBIOBANK/3ddata/:/data/ \
  --command -- bash /project/src/bash/start_script.sh \
    python3 /project/src/python/training/train_aekl.py \
      seed=${seed} \
      run_dir=${run_dir} \
      training_ids=${training_ids} \
      validation_ids=${validation_ids} \
      config_file=${config_file} \
      batch_size=${batch_size} \
      n_epochs=${n_epochs} \
      eval_freq=${eval_freq} \
      num_workers=${num_workers} \
      use_bfloat=${use_bfloat} \
      experiment=${experiment}

seed=42
run_dir="aekl_v1"
training_ids="/project/outputs/ids/training.tsv"
validation_ids="/project/outputs/ids/validation.tsv"
config_file="/project/configs/stage1/aekl_v1.yaml"
batch_size=4
n_epochs=500
eval_freq=10
num_workers=32
use_bfloat=0
experiment="AEKL"

runai submit \
  --name s2-aekl-v1 \
  --image aicregistry:5000/wds20:ldms2 \
  --backoff-limit 0 \
  --gpu 4 \
  --cpu 16 \
  --large-shm \
  --run-as-user \
  --node-type "A100" \
  --host-ipc \
  --project wds20 \
  --volume /nfs/home/wds20/projects/generative_cardiac/:/project/ \
  --volume /nfs/home/wds20/datasets/CARDIAC_UKBIOBANK/3ddata/:/data/ \
  --command -- bash /project/src/bash/start_script.sh \
    python3 /project/src/python/training/train_aekl.py \
      seed=${seed} \
      run_dir=${run_dir} \
      training_ids=${training_ids} \
      validation_ids=${validation_ids} \
      config_file=${config_file} \
      batch_size=${batch_size} \
      n_epochs=${n_epochs} \
      eval_freq=${eval_freq} \
      num_workers=${num_workers} \
      use_bfloat=${use_bfloat} \
      experiment=${experiment}

# GENERATES NANS THAT DESTROY THE IMAGE
#seed=42
#run_dir="aekl_v2"
#training_ids="/project/outputs/ids/training.tsv"
#validation_ids="/project/outputs/ids/validation.tsv"
#config_file="/project/configs/stage1/aekl_v2.yaml"
#batch_size=4
#n_epochs=200
#eval_freq=10
#num_workers=16
#use_bfloat=0
#experiment="AEKL"
#
#runai submit \
#  --name s2-aekl-v2 \
#  --image aicregistry:5000/wds20:ldms2 \
#  --backoff-limit 0 \
#  --gpu 4 \
#  --cpu 16 \
#  --large-shm \
#  --run-as-user \
#  --host-ipc \
#  --project wds20 \
#  --volume /nfs/home/wds20/projects/generative_cardiac/:/project/ \
#  --volume /nfs/home/wds20/datasets/CARDIAC_UKBIOBANK/3ddata/:/data/ \
#  --command -- bash /project/src/bash/start_script.sh \
#    python3 /project/src/python/training/train_aekl.py \
#      seed=${seed} \
#      run_dir=${run_dir} \
#      training_ids=${training_ids} \
#      validation_ids=${validation_ids} \
#      config_file=${config_file} \
#      batch_size=${batch_size} \
#      n_epochs=${n_epochs} \
#      eval_freq=${eval_freq} \
#      num_workers=${num_workers} \
#      use_bfloat=${use_bfloat} \
#      experiment=${experiment}


# TOO UNSTABLE ON DGX CLUSTER
#seed=42
#run_dir="aekl_v3"
#training_ids="/project/outputs/ids/training.tsv"
#validation_ids="/project/outputs/ids/validation.tsv"
#config_file="/project/configs/stage1/aekl_v3.yaml"
#batch_size=4
#n_epochs=200
#eval_freq=10
#num_workers=16
#use_bfloat=0
#experiment="AEKL"
#
#runai submit \
#  --name s2-aekl-v3 \
#  --image aicregistry:5000/wds20:ldms2 \
#  --backoff-limit 0 \
#  --gpu 4 \
#  --cpu 16 \
#  --large-shm \
#  --run-as-user \
#  --host-ipc \
#  --project wds20 \
#  --volume /nfs/home/wds20/projects/generative_cardiac/:/project/ \
#  --volume /nfs/home/wds20/datasets/CARDIAC_UKBIOBANK/3ddata/:/data/ \
#  --command -- bash /project/src/bash/start_script.sh \
#    python3 /project/src/python/training/train_aekl.py \
#      seed=${seed} \
#      run_dir=${run_dir} \
#      training_ids=${training_ids} \
#      validation_ids=${validation_ids} \
#      config_file=${config_file} \
#      batch_size=${batch_size} \
#      n_epochs=${n_epochs} \
#      eval_freq=${eval_freq} \
#      num_workers=${num_workers} \
#      use_bfloat=${use_bfloat} \
#      experiment=${experiment}

# TOO UNSTABLE ON DGX CLUSTER
#seed=42
#run_dir="aekl_v4"
#training_ids="/project/outputs/ids/training.tsv"
#validation_ids="/project/outputs/ids/validation.tsv"
#config_file="/project/configs/stage1/aekl_v4.yaml"
#batch_size=4
#n_epochs=200
#eval_freq=10
#num_workers=16
#use_bfloat=0
#experiment="AEKL"
#
#runai submit \
#  --name s2-aekl-v4 \
#  --image aicregistry:5000/wds20:ldms2 \
#  --backoff-limit 0 \
#  --gpu 4 \
#  --cpu 16 \
#  --large-shm \
#  --run-as-user \
#  --host-ipc \
#  --project wds20 \
#  --volume /nfs/home/wds20/projects/generative_cardiac/:/project/ \
#  --volume /nfs/home/wds20/datasets/CARDIAC_UKBIOBANK/3ddata/:/data/ \
#  --command -- bash /project/src/bash/start_script.sh \
#    python3 /project/src/python/training/train_aekl.py \
#      seed=${seed} \
#      run_dir=${run_dir} \
#      training_ids=${training_ids} \
#      validation_ids=${validation_ids} \
#      config_file=${config_file} \
#      batch_size=${batch_size} \
#      n_epochs=${n_epochs} \
#      eval_freq=${eval_freq} \
#      num_workers=${num_workers} \
#      use_bfloat=${use_bfloat} \
#      experiment=${experiment}


seed=42
run_dir="aekl_v5"
training_ids="/project/outputs/ids/training.tsv"
validation_ids="/project/outputs/ids/validation.tsv"
config_file="/project/configs/stage1/aekl_v5.yaml"
batch_size=4
n_epochs=500
eval_freq=10
num_workers=32
use_bfloat=0
experiment="AEKL"

runai submit \
  --name s2-aekl-v5 \
  --image aicregistry:5000/wds20:ldms2 \
  --backoff-limit 0 \
  --gpu 4 \
  --cpu 16 \
  --large-shm \
  --run-as-user \
  --node-type "A100" \
  --host-ipc \
  --project wds20 \
  --volume /nfs/home/wds20/projects/generative_cardiac/:/project/ \
  --volume /nfs/home/wds20/datasets/CARDIAC_UKBIOBANK/3ddata/:/data/ \
  --command -- bash /project/src/bash/start_script.sh \
    python3 /project/src/python/training/train_aekl.py \
      seed=${seed} \
      run_dir=${run_dir} \
      training_ids=${training_ids} \
      validation_ids=${validation_ids} \
      config_file=${config_file} \
      batch_size=${batch_size} \
      n_epochs=${n_epochs} \
      eval_freq=${eval_freq} \
      num_workers=${num_workers} \
      use_bfloat=${use_bfloat} \
      experiment=${experiment}

# ------------------------------------------------------------------
# Highres
# ------------------------------------------------------------------

seed=42
run_dir="highres_aekl_v0"
training_ids="/project/outputs/ids/training.tsv"
validation_ids="/project/outputs/ids/validation.tsv"
config_file="/project/configs/stage1/highres_aekl_v0.yaml"
batch_size=4
n_epochs=500
eval_freq=10
num_workers=32
use_bfloat=0
experiment="AEKL"

runai submit \
  --name s2-highres-aekl-v0 \
  --image aicregistry:5000/wds20:ldms2 \
  --backoff-limit 0 \
  --gpu 4 \
  --cpu 16 \
  --large-shm \
  --run-as-user \
  --node-type "A100" \
  --host-ipc \
  --project wds20 \
  --volume /nfs/home/wds20/projects/generative_cardiac/:/project/ \
  --volume /nfs/home/wds20/datasets/CARDIAC_UKBIOBANK/3ddata/:/data/ \
  --command -- bash /project/src/bash/start_script.sh \
    python3 /project/src/python/training/train_aekl.py \
      seed=${seed} \
      run_dir=${run_dir} \
      training_ids=${training_ids} \
      validation_ids=${validation_ids} \
      config_file=${config_file} \
      batch_size=${batch_size} \
      n_epochs=${n_epochs} \
      eval_freq=${eval_freq} \
      num_workers=${num_workers} \
      use_bfloat=${use_bfloat} \
      experiment=${experiment}


seed=42
run_dir="highres_aekl_v1"
training_ids="/project/outputs/ids/training.tsv"
validation_ids="/project/outputs/ids/validation.tsv"
config_file="/project/configs/stage1/highres_aekl_v1.yaml"
batch_size=4
n_epochs=500
eval_freq=10
num_workers=32
use_bfloat=0
experiment="AEKL"

runai submit \
  --name s2-highres-aekl-v1 \
  --image aicregistry:5000/wds20:ldms2 \
  --backoff-limit 0 \
  --gpu 4 \
  --cpu 16 \
  --large-shm \
  --run-as-user \
  --node-type "A100" \
  --host-ipc \
  --project wds20 \
  --volume /nfs/home/wds20/projects/generative_cardiac/:/project/ \
  --volume /nfs/home/wds20/datasets/CARDIAC_UKBIOBANK/3ddata/:/data/ \
  --command -- bash /project/src/bash/start_script.sh \
    python3 /project/src/python/training/train_aekl.py \
      seed=${seed} \
      run_dir=${run_dir} \
      training_ids=${training_ids} \
      validation_ids=${validation_ids} \
      config_file=${config_file} \
      batch_size=${batch_size} \
      n_epochs=${n_epochs} \
      eval_freq=${eval_freq} \
      num_workers=${num_workers} \
      use_bfloat=${use_bfloat} \
      experiment=${experiment}


# GENERATES NANS THAT DESTROY THE IMAGE
#seed=42
#run_dir="highres_aekl_v2"
#training_ids="/project/outputs/ids/training.tsv"
#validation_ids="/project/outputs/ids/validation.tsv"
#config_file="/project/configs/stage1/highres_aekl_v2.yaml"
#batch_size=4
#n_epochs=200
#eval_freq=10
#num_workers=32
#use_bfloat=0
#experiment="AEKL"
#
#runai submit \
#  --name s2-highres-aekl-v2 \
#  --image aicregistry:5000/wds20:ldms2 \
#  --backoff-limit 0 \
#  --gpu 4 \
#  --cpu 16 \
#  --large-shm \
#  --run-as-user \
#  --host-ipc \
#  --project wds20 \
#  --volume /nfs/home/wds20/projects/generative_cardiac/:/project/ \
#  --volume /nfs/home/wds20/datasets/CARDIAC_UKBIOBANK/3ddata/:/data/ \
#  --command -- bash /project/src/bash/start_script.sh \
#    python3 /project/src/python/training/train_aekl.py \
#      seed=${seed} \
#      run_dir=${run_dir} \
#      training_ids=${training_ids} \
#      validation_ids=${validation_ids} \
#      config_file=${config_file} \
#      batch_size=${batch_size} \
#      n_epochs=${n_epochs} \
#      eval_freq=${eval_freq} \
#      num_workers=${num_workers} \
#      use_bfloat=${use_bfloat} \
#      experiment=${experiment}


# TOO UNSTABLE ON DGX CLUSTER
#seed=42
#run_dir="highres_aekl_v3"
#training_ids="/project/outputs/ids/training.tsv"
#validation_ids="/project/outputs/ids/validation.tsv"
#config_file="/project/configs/stage1/highres_aekl_v3.yaml"
#batch_size=4
#n_epochs=200
#eval_freq=10
#num_workers=32
#use_bfloat=0
#experiment="AEKL"
#
#runai submit \
#  --name s2-highres-aekl-v3 \
#  --image aicregistry:5000/wds20:ldms2 \
#  --backoff-limit 0 \
#  --gpu 4 \
#  --cpu 16 \
#  --large-shm \
#  --run-as-user \
#  --host-ipc \
#  --project wds20 \
#  --volume /nfs/home/wds20/projects/generative_cardiac/:/project/ \
#  --volume /nfs/home/wds20/datasets/CARDIAC_UKBIOBANK/3ddata/:/data/ \
#  --command -- bash /project/src/bash/start_script.sh \
#    python3 /project/src/python/training/train_aekl.py \
#      seed=${seed} \
#      run_dir=${run_dir} \
#      training_ids=${training_ids} \
#      validation_ids=${validation_ids} \
#      config_file=${config_file} \
#      batch_size=${batch_size} \
#      n_epochs=${n_epochs} \
#      eval_freq=${eval_freq} \
#      num_workers=${num_workers} \
#      use_bfloat=${use_bfloat} \
#      experiment=${experiment}


# TOO UNSTABLE ON DGX CLUSTER
#seed=42
#run_dir="highres_aekl_v4"
#training_ids="/project/outputs/ids/training.tsv"
#validation_ids="/project/outputs/ids/validation.tsv"
#config_file="/project/configs/stage1/highres_aekl_v4.yaml"
#batch_size=4
#n_epochs=200
#eval_freq=10
#num_workers=32
#use_bfloat=0
#experiment="AEKL"
#
#runai submit \
#  --name s2-highres-aekl-v4 \
#  --image aicregistry:5000/wds20:ldms2 \
#  --backoff-limit 0 \
#  --gpu 4 \
#  --cpu 16 \
#  --large-shm \
#  --run-as-user \
#  --host-ipc \
#  --project wds20 \
#  --volume /nfs/home/wds20/projects/generative_cardiac/:/project/ \
#  --volume /nfs/home/wds20/datasets/CARDIAC_UKBIOBANK/3ddata/:/data/ \
#  --command -- bash /project/src/bash/start_script.sh \
#    python3 /project/src/python/training/train_aekl.py \
#      seed=${seed} \
#      run_dir=${run_dir} \
#      training_ids=${training_ids} \
#      validation_ids=${validation_ids} \
#      config_file=${config_file} \
#      batch_size=${batch_size} \
#      n_epochs=${n_epochs} \
#      eval_freq=${eval_freq} \
#      num_workers=${num_workers} \
#      use_bfloat=${use_bfloat} \
#      experiment=${experiment}


seed=42
run_dir="highres_aekl_v5"
training_ids="/project/outputs/ids/training.tsv"
validation_ids="/project/outputs/ids/validation.tsv"
config_file="/project/configs/stage1/highres_aekl_v5.yaml"
batch_size=4
n_epochs=500
eval_freq=10
num_workers=32
use_bfloat=0
experiment="AEKL"

runai submit \
  --name s2-highres-aekl-v5 \
  --image aicregistry:5000/wds20:ldms2 \
  --backoff-limit 0 \
  --gpu 4 \
  --cpu 16 \
  --large-shm \
  --run-as-user \
  --node-type "A100" \
  --host-ipc \
  --project wds20 \
  --volume /nfs/home/wds20/projects/generative_cardiac/:/project/ \
  --volume /nfs/home/wds20/datasets/CARDIAC_UKBIOBANK/3ddata/:/data/ \
  --command -- bash /project/src/bash/start_script.sh \
    python3 /project/src/python/training/train_aekl.py \
      seed=${seed} \
      run_dir=${run_dir} \
      training_ids=${training_ids} \
      validation_ids=${validation_ids} \
      config_file=${config_file} \
      batch_size=${batch_size} \
      n_epochs=${n_epochs} \
      eval_freq=${eval_freq} \
      num_workers=${num_workers} \
      use_bfloat=${use_bfloat} \
      experiment=${experiment}
