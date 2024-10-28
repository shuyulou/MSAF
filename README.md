## MSAF : Multimodal Sentiment Detection via Multi-Scale Adaptive Fusion

This is the repository of MSAF : Multimodal Sentiment Detection via Multi-Scale Adaptive Fusion

![image](MSAF.png)

### Requirements:

```shell
numpy==1.24.3
scikit_learn==1.3.2
pytorch==2.0.0
```

### Download:

MVSA: https://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/

HFM: https://github.com/headacheboy/data-of-multimodal-sarcasm-detection

RU-Senti: https://github.com/PhenoixYANG/TOM

Checkpoint of BERT-base: https://huggingface.co/google-bert/bert-base-uncased

Checkpoint of swin_transformer: https://github.com/microsoft/Swin-Transformer

### Multimodal Knowledge Graph Completion:

train on MVSA-Single

```shell
sh train_single.sh 
```

train on MVSA-Multiple

```shell
sh train_mul.sh 
```

train on HFM

```shell
sh train_hfm.sh 
```

train on RU-Senti

```shell
sh train_ru.sh 
```
