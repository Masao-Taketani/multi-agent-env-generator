python enc_dec_training.py \
    --log_dir encdec_results/carla \
    --use_perceptual_loss \
    --batch 64 \
    --data_path ./datasets/carla/data1,./datasets/carla/data2,./datasets/carla/data3,./datasets/carla/data4,./datasets/carla/data5,./datasets/carla/data6 \
    --dataset carla \
    --nfilterDec 48