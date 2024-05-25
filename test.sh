python main.py \
--model mobilenet_v3_small \
--eval true \
--batch_size 64 \
--finetune ckpt0523/checkpoint-best.pth \
--input_size 800 \
--nb_classes 2 \
--data_set image_folder \
--data_path /home/cw/Desktop/liu/dataset/MobileNetv3/train \
--eval_data_path  /home/cw/Desktop/liu/dataset/MobileNetv3/val \
--output_dir ./0523_test
# --data_set image_folder \