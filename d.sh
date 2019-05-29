
for i in {49..63}
do
  CUDA_VISIBLE_DEVICES=3 python ./bin/dlib_reproduce --study=supervised_vae_study_v1 --model_num=${i} > output/${i}.cout 2> output/${i}.cerr
done

