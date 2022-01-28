
VID=$1; CUDA_VISIBLE_DEVICES=0 python train.py \
  --vid $VID \
  --exp_name rel/$VID \
  --train_ratio 1 --num_epochs 10
