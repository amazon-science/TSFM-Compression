# Understanding Transformers for Time Series: Rank Structure, Flow-of-ranks, and Compressibility

This repository is associated with the paper "Understanding Transformers for Time Series: Rank Structure, Flow-of-ranks, and Compressibility" by Annan Yu, Danielle C. Maddix, Boran Han, Xiyuan Zhang, Abdul Fatir Ansari, Oleksandr Shchur, Christos Faloutsos, Andrew Gordon Wilson, Michael W. Mahoney, and Yuyang Wang. It contains codes that are used to investigate the role of a low-rank structure in time-series foundation models and how it can be used to compress the models.

# Pretraining Compressed Models

To pretrain a compressed Chronos model, follow the following steps:

### 1. Download training repository and data

   One needs to first clone the [`Chronos repository`](https://github.com/amazon-science/chronos-forecasting). Then, following the instruction in the repository to download the training data.

### 2. Replace the T5 source file

   In order to compress the model for pretraining, we need to modify the T5 source file. In your python environment, find the T5 source file (e.g., `lib64/python3.11/site-packages/transformers/models/t5/modeling_t5.py`) and replace its content with the content of [`modeling_t5_dense_flowofranks.py`](./T5-variants/modeling_t5_dense_flowofranks.py) in this repository. Leave the name of `modeling_t5.py` unchanged.
   
   Next, replace the configuration file (e.g., `lib64/python3.11/site-packages/transformers/models/t5/configuration_t5.py`) with [`configuration_t5.py`](./T5-variants/configuration_t5.py) in this repository.

**Note:** We assume that version `4.49.0` of the `transformers` package is used. Compatibility issues may arise otherwise.

### 3. Complete the configuration files

   Replace the configuration files with one of the following files: [`chronos-t5-0_25.yaml`](./configs/chronos-t5-0_25.yaml), [`chronos-t5-0_15.yaml`](./configs/chronos-t5-0_15.yaml), and [`chronos-t5-0_075.yaml`](./configs/chronos-t5-0_075.yaml), which achieve a compression-to ratio of 25%, 15%, and 7.5%, respectively, and set `training_data_paths` in your configuration file to paths to your training data.

### 4. Training and evaluation

You can now pretrain and evaluate your compressed TSFM by following in the instructions in the [`Chronos repository`](https://github.com/amazon-science/chronos-forecasting).

# Evaluation

[`evaluation-scripts`](./evaluation-scripts/): contains the script to evaluate a normally pretrained TSFM that is compressed post-training. It is used to produce Table 2 of our paper.

# Compare TSFMs to LLMs

[`comparison`](./comparison/): contains two standalone evaluation scripts that we used to evaluate and compare Chronos models and T5 large language models compressed after pretraining. These are used to produce Table 2 of our paper.

# Source

This repository contains modified versions of the code found in the following repositories:

[**chronos-forecasting**](https://github.com/amazon-science/chronos-forecasting): For training compressed Chronos models that are used to test the compressibility of univariate time-series foundation models based on the rank structures of time-series inputs.

[**transformers**](https://github.com/huggingface/transformers): For building a compressed TSFM's T5 Transformer backbone.

# Citation

If you use this code, or our work, please cite:

```
@inproceedings{yu2026rank,
    title={Understanding Transformers for Time Series: Rank Structure, Flow-of-ranks, and Compressibility},
    author={Yu, A. and Maddix, D.C. and Han, B. and Zhang, X. and Ansari, A.F. and Shchur, O. and Faloutsos, C. and Wilson, A.G. and Mahoney, M.W. and Wang, Y.},
    booktitle={International Conference on Learning Representations},
    year={2026}
}
```

# License

This project is licensed under the Apache-2.0 License.