export PYTHONPATH=/home/djlee/icefall:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="7"

./pruned_transducer_stateless7/decode.py \
    --epoch 30 \
    --avg 1 \
    --use-averaged-model false \
    --exp-dir ./pruned_transducer_stateless7/icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11/exp \
    --feedforward-dims  "1024,1024,2048,2048,1024" \
    --max-duration 500 \
    --decoding-method greedy_search
