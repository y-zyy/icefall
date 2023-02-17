#!/usr/bin/env bash
export PYTHONPATH=/home/djlee/icefall:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="7"

./pruned_transducer_stateless7/pretrained.py \
  --checkpoint ./pruned_transducer_stateless7/icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11/exp/pretrained.pt \
  --bpe-model ./data/lang_bpe_500/bpe.model \
  --feedforward-dims  "1024,1024,2048,2048,1024" \
  --method greedy_search \
  ./test.flac