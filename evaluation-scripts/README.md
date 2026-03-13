## Evaluate the In-Domain and Zero-Shot Performance of Compressed Chronos Models

To compute the in-domain and zero-shot WQL and MASE of the compressed Chronos models, one needs to first download the [`public chronos-forecasting repository`](https://github.com/amazon-science/chronos-forecasting/tree/main). Replace the file `chronos-forecasting/scripts/evaluation/evaluate.py` with [`evaluate.py`](./evaluate.py) in this directory. Then, one can evaluate the compressed pretrained Chronos models by calling

```
python evaluation/evaluate.py evaluation/configs/in-domain.yaml evaluation/results/chronos-t5-small-in-domain.csv \
    --chronos-model-id "amazon/chronos-t5-small" \
    --batch-size=32 \
    --device=cuda:0 \
    --num-samples 20 \
    --epsilon=0.01
```

and

```
python evaluation/evaluate.py evaluation/configs/zero-shot.yaml evaluation/results/chronos-t5-small-zero-shot.csv \
    --chronos-model-id "amazon/chronos-t5-small" \
    --batch-size=32 \
    --device=cuda:0 \
    --num-samples 20 \
    --epsilon=0.01
```

Here, `epsilon` is the threshold that we use to truncate the singular values. See section 5 of the paper for more details.