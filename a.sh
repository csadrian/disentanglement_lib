#134
for i in {0..15}
do
  CUDA_VISIBLE_DEVICES=0 dlib_reproduce --study=supervised_vae_study_v1 --model_num=${i} > output/${i}.cout 2> output/${i}.cerr
done

