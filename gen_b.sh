
for i in {16..31}
do
  CUDA_VISIBLE_DEVICES=1 dlib_reproduce --study=generalization_vae_study_v1 --model_num=${i} > output/${i}.cout 2> output/${i}.cerr
done

