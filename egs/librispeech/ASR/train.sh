export PYTHONPATH=/home/djlee/icefall:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="4,5,6,7"

./pruned_transducer_stateless7/train.py \
	--world-size 4 \
	--num-epochs 30 \
	--full-libri 1 \
	--use-fp16 1 \
	--max-duration 750 \
	--exp-dir pruned_transducer_stateless7/exp \
	--feedforward-dims  "1024,1024,2048,2048,1024" \
	
