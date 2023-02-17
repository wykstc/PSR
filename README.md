# PSR

This is the official implementation for the paper "EMP: Emotion-guided Multi-modal Fusion and Contrastive
Learning for Personality Traits Recognition".

# Structure of MERC
![image](structure.png)

## Table of Contents

- [Dependencies](#security)
- [Requirement](#background)
- [Install](#install)
- [Dataset](#dataset)
- [Usage](#usage)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
- [License](#license)

## Dependencies

- Python 3.9.1
- PyTorch toolbox (1.12.1+cu113)
- Linux 5.11.0-46-generic

Please follow the paper to pre-process the data. 

## Requirement
- We use PyG (PyTorch Geometric) for the GNN component in our architecture. [RGCNConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.RGCNConv) and [TransformerConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.TransformerConv)
- We use PyGCL for GCL (Graph Contrastive Learning) network in out framework. [PyGCL](https://github.com/PyGCL/PyGCL)
- We use sentence transfomer for text feature extraction. [Sentence Embedding](https://huggingface.co/sentence-transformers/paraphrase-distilroberta-base-v1)


## Install
To easily reproduce our results, you can install the environments by
```
pip install -r requirements.txt
```


## Dataset
Chalearn first impressions dataset can be found in [First impressions](https://chalearnlap.cvc.uab.cat/dataset/24/description/)

The ELEA dataset can be found on this official website and you need to apply it
[ELEA](https://www.idiap.ch/en/dataset/elea) 


## Usage

### Train the model

```
ulimit -SHn 51200
python main.py --accelerator 'gpu' --devices 1  
```

### Evaluate the model, you need to change model to

```
trainer.test(model, data_module)
```
