runai submit \
  --name zip-mimic \
  --image aicregistry:5000/wds20:mimic_zipping \
  --backoff-limit 0 \
  --gpu 0 \
  --cpu 4 \
  --large-shm \
  --run-as-user \
  --host-ipc \
  --project wds20 \
  --volume /nfs/home/wds20/datasets/MIMIC-CXR-JPG_v2.0.0/:/sourcedata/ \
  --volume /nfs/home/wds20/projects/generative_mimic/:/project/ \
  --command -- sleep infinity


