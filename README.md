# E2VD

The official code repository of "E2VD: a unified evolution-driven framework for virus variation drivers prediction".


# Demo and Instructions for Use

## Quick start

We provide a quick demo for the binding affinity prediction experiment to show the learning and predicting ability of E2VD. The dataset used here is single-site mutation benchmark.

- Training

Download the extracted sequence feature of the training and testing data.

```shell
cd demo_bind/data
wget --no-check-certificate https://figshare.com/ndownloader/files/46588753 -O QuickStart.zip
unzip QuickStart.tar.gz
cd ..
```

Run `bind_train_5fold.py` to do 5 fold cross validation using training data.

```shell
python bind_train_5fold.py > train_output.txt 2>&1
```

[<img src="https://colab.research.google.com/assets/colab-badge.svg">](https://colab.research.google.com/github/ZhiweiNiepku/E2VD/blob/main/examples/E2VD_QuickStart_train.ipynb)

- Testing

Use training result to predict the binding affinity of the blind test set. Besides, we provide our training checkpoints to make a quick prediction. 

```shell
# run prediction
python bind_predict_5fold.py > test_result.txt 2>&1
```

[<img src="https://colab.research.google.com/assets/colab-badge.svg">](https://colab.research.google.com/github/ZhiweiNiepku/E2VD/blob/main/examples/E2VD_QuickStart_predict.ipynb)

The expected outputs are as follows.

| AU-ROC   | Acc   | F1-Score | Precision | Recall | MSE   | Corr  |
| ----- | ----- | -------- | --------- | ------ | ----- | ----- |
| 92.98 | 91.11 | 91.58    | 87.43     | 96.30  | 0.058 | 0.870 |

It takes only several hours to train on training data, and several seconds to predict on test data.

## Protein sequence feature extraction

**!! NOTE: We provide the extracted features of the sequences used for experiments at the end of this section. You can use the extracted results directly for downstream applications and skip this section.**

Use our pretrained backbone to extract feature embeddings of protein sequences for downstream application. The pretrained checkpoint can be downloaded in the table below.

Use `pretraining/extract_embedding_local.py` to extract feature embeddings on your computer or server. Please modify the file path in the code before executing.

```python
    ms.context.set_context(variable_memory_max_size="30GB")
    ckpt_path = '<path to the pretrained model>/checkpoint.ckpt'

    data_path = '<path to the dir to extract features>/raw/'
    filelist = glob.glob(data_path+"*.txt")
    filelist.sort()
    result_path = "<path to the dir to save results>/result/"
    os.makedirs(result_path, exist_ok=True)
```

We provide the extracted protein features as well as the pretrained checkpoints here for the convenient experiments on downstream tasks.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10043360.svg)](https://zenodo.org/doi/10.5281/zenodo.10038909)

| Section                                  | Download Link                                                | Content                                                      |
| ---------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Pretrained checkpoint**                | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10043360.svg)](https://zenodo.org/records/10043360/files/checkpoint_protfound_v.ckpt) | Our pretrained checkpoint.                                    |
| **QuickStart**                           | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10043360.svg)](https://zenodo.org/records/10043360/files/QuickStart.tar.gz) | Extracted features and trained checkpoints.                  |
| **Single-site mutation benchmark**       | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10043360.svg)](https://zenodo.org/records/10043360/files/Single-site_mutation_benchmark.tar.gz) | Raw RBD sequences and extracted features.                    |
| **Multi-site mutation benchmark**        | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10043360.svg)](https://zenodo.org/records/10043360/files/Multi-site_mutation_benchmark.tar.gz) | Raw RBD sequences and extracted features.                    |
| **Determining high-risk mutation sites** | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10043360.svg)](https://zenodo.org/records/10043360/files/Determining_high-risk_mutation_sites.tar.gz) | Raw protein sequences, extracted features, and trained checkpoints. |
