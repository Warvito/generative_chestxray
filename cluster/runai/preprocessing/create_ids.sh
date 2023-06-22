metadata_path="/project/src/data/mimic-cxr-2.0.0-metadata.csv"
output_dir="/project/outputs/ids/"

runai submit \
  --name organise-mimic \
  --image aicregistry:5000/wds20:ldm_mimic \
  --backoff-limit 0 \
  --gpu 0 \
  --cpu 4 \
  --large-shm \
  --run-as-user \
  --host-ipc \
  --project wds20 \
  --volume /nfs/home/wds20/datasets/MIMIC-CXR-JPG_v2.0.0/rawdata:/rawdata/ \
  --volume /nfs/home/wds20/datasets/MIMIC-CXR-JPG_v2.0.0/physionet.org/files/mimic-cxr-jpg/2.0.0/:/sourcedata/ \
  --volume /nfs/home/wds20/projects/generative_mimic/:/project/ \
  --command -- bash /project/src/bash/start_script.sh \
      python3 /project/src/python/preprocessing/create_ids.py \
          metadata_path=${metadata_path} \
          output_dir=${output_dir}
