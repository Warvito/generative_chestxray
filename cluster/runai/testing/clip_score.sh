output_dir="/project/outputs/metrics/v0.3/guidance_scale_1.0/"

counter=0
for guidance_scale in 1.0 1.5 1.75 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 ; do
    samples_dir="/project/outputs/samples_fid_v03/guidance_scale_${guidance_scale}"
    runai submit \
      --name  mimic-clip-score-${counter}  \
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
        python3 /project/src/python/testing/compute_clip_score.py \
          samples_dir=${samples_dir} \
          output_dir=${output_dir}
    counter=$((counter+1))
    sleep 3
done
