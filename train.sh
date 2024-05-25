python main.py \
--model mobilenet_v3_small \
--input_size 800 \
--nb_classes 2 \
--data_set image_folder \
--finetune 450_act3_mobilenetv3_small.pth \
--epochs 300 --batch_size 64 --lr 4e-3 --update_freq 2 --model_ema false --model_ema_eval false --use_amp true \
--data_path /home/cw/Desktop/liu/dataset/MobileNetv3/train \
--eval_data_path /home/cw/Desktop/liu/dataset/MobileNetv3/val \
--output_dir ./ckpt0523 \
> out0523.log 

# --enable_wandb true --project mobilenetv3