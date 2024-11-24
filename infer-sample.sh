CUDA_VISIBLE_DEVICES=xxx python test.py \
--config landscape-rec \
--model_config InfinityGAN \
--input_path /path/of/images  \
--gan_model_path "/path/of/ckpts/scenen_best_fid.pth.tar" \
--save_path /path/to/exp/ \
--checkpoint "/path/to/exp/xx.ckpt" \
--scale 0 \
--attribute None