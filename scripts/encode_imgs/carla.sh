encdec_ckpt=$1

python latent_encoder.py \
    --ckpt ${encdec_ckpt} \
    --results_path encoded_dataset/carla \
    --data_path datasets/carla/data1,datasets/carla/data2,datasets/carla/data3,datasets/carla/data4,datasets/carla/data5,datasets/carla/data6 \
    --dataset carla \
    --img_size 64x64