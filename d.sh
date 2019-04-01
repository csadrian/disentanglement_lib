
for i in {49..63}
do
  CUDA_VISIBLE_DEVICES=3 dlib_reproduce --study=augmented_variance_vae_study_v1 --model_num=${i} > output/${i}.cout 2> output/${i}.cerr
done

