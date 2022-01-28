
CKP=ckpts/$1
VID=$2
EXP=$3
OUT=$4
MASKS_N_SAMPLES=$5
SUMMARY_N_SAMPLES=$6

EPOCH=9

CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --path $CKP\/$VID\/epoch\=$EPOCH\.ckpt \
  --vid $VID --exp $EXP \
  --is_eval_script \
  --outputs $OUT \
  --masks_n_samples $MASKS_N_SAMPLES \
  --summary_n_samples $SUMMARY_N_SAMPLES
