export PYTHONPATH=/home/djlee/icefall:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="4,5,6,7"

./pruned_transducer_stateless7/train.py \
   --world-size 4 \
   --num-epochs 30 \
   --start-epoch 1 \
   --use-fp16 1 \
   --exp-dir pruned_transducer_stateless7/exp_spk_cond_1 \
   --full-libri 1 \
   --max-duration 280
