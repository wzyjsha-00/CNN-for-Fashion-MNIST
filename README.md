# CNN on Fashion-MNIST

## Introduction  
This repository is the reproduction of some classical Convolutional Neural Networks on [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) as followed.
Please feel free to ask anything!
- [**LeNet**](model/LeNet.py), [*Paper Link*](https://ieeexplore.ieee.org/document/726791)

## Baseline Environment and Using
```buildoutcfg
python 3.9.11
cuda 11.0.3
cudnn 8.0.5
torch 1.7.1
```
- **Install pytorch:** ```pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html```
- **Install dependencies:** ```pip install -r requirements.txt```
- **Download Data:** get [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) then put into the above folder
- **Check** the [hyper parameter](config/hyper_param.py)

## References
1. Y. Lecun, L. Bottou, Y. Bengio and P. Haffner, "Gradient-based learning applied to document recognition," in Proceedings of the IEEE, vol. 86, no. 11, pp. 2278-2324, Nov. 1998, doi: 10.1109/5.726791.


##  Working Record
|        State        |Networks|Acc|Model Param|
|:-------------------:|:------:|---|-----------|
|&#9745; 09 June, 2022|[LeNet](model/LeNet.py)|||
|&#9744;              |[AlexNet]()||
|&#9744;              |[VGGNet]()||
|&#9744;              |[InceptionNet]()||
|&#9744;              |[ResNet]()||
