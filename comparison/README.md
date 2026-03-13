## Compute the perplexity and Jaccard overlaps of compressed T5

The experiment to compare the perplexity and Jaccard overlaps of compressed T5 models can be run using the following command:

```
python compress_T5.py \
   --num-sentences 1000 \
   --batch-size 8 \
   --topk 10
```

Here, `topk` is the index to be used in computing the Jaccard overlaps, and `num-sentences` is the number of evaluation sentences over which the metrics will be averaged. The results will be saved to the [`results`](./results/) directory.

## Compute the Jaccard overlaps of compressed T5

The experiment to compare the Jaccard overlaps of compressed T5 models can be run using the following command:

```
python compress_chronos.py \
   --dataset electricity_15min \
   --num-series 1000 \
   --series-len 512
```

Here, `topk` is the index to be used in computing the Jaccard overlaps, `num-series` is the number of evaluation time-series over which the metric will be averaged, and `dataset` is the autogluon dataset from which the time-series are sampled from. The available datasets can be found on [huggingface](https://huggingface.co/datasets/autogluon/chronos_datasets). The results will be saved to the [`results`](./results/) directory.