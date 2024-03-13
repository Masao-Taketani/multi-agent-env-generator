encdec_ckpt=$1

python latent_encoder.py \
    --ckpt ${encdec_ckpt} \
    --results_path encoded_dataset/pong \
    --data_path datasets/pong/ \
    --dataset pong \
    --img_size 64x64