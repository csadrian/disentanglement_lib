
for i in {80..95}
do
  CUDA_VISIBLE_DEVICES=5 dlib_reproduce --study=supervised_vae_study_v1 --model_num=${i} > output/${i}.cout 2> output/${i}.cerr
done

