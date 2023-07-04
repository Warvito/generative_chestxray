seed=42
test_ids="/project/outputs/ids/test.tsv"
num_workers=8
batch_size=16

counter=0
for guidance_scale in 1.0 1.5 1.75 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 ; do
    sample_dir="/project/outputs/samples_fid_v03/guidance_scale_${guidance_scale}"
    runai submit \
      --name  mimic-fid-${counter} \
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
        python3 /project/src/python/testing/compute_fid.py \
          seed=${seed} \
          sample_dir=${sample_dir} \
          test_ids=${test_ids} \
          batch_size=${batch_size} \
          num_workers=${num_workers}
    counter=$((counter+1))
    sleep 3
done
