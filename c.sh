
for i in {32..48}
do
  CUDA_VISIBLE_DEVICES=2 dlib_reproduce --study=augmented_variance_vae_study_v1 --model_num=${i} > output/${i}.cout 2> output/${i}.cerr
done

