
for i in {80..89}
do
  CUDA_VISIBLE_DEVICES=5 python ./bin/dlib_reproduce --study=generalization_vae_study_v1 --model_num=${i} > output/${i}.cout 2> output/${i}.cerr
done

