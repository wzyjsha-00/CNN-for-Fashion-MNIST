# CNN on Fashion-MNIST

## Introduction  
This repository is the reproduction of some classical **C**onvolutional **N**eural **N**etworks on [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) as followed.
Please feel free to ask anything!
- [**LeNet**](model/LeNet.py), [*Paper Link*](https://ieeexplore.ieee.org/document/726791)
- [**AlexNet**](model/AlexNet.py), [*Paper Link*](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf)
- [**VGGNet**](model/VGGNet.py), [*Paper Link*](https://arxiv.org/abs/1409.1556)

## Baseline Environment and Setup
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


##  Working Record
|        State        |Network|Train Epochs / Time|Best Test Acc|Model Param|
|:-------------------:|:-----:|:-----------------:|:-----------:|:---------:|
|&#9745; 09 June, 2022|[LeNet](model/LeNet.py)|30 / 18m-29s|0.9016|[LeNet-best.pth.tar](model/LeNet5_Pretrained/best.pth.tar)|
|&#9745; 11 June, 2022|[AlexNet<sup>1](model/AlexNet.py)|30 / 35m-39s|0.9219|[AlexNet-best.pth.tar](model/AlexNet_Pretrained/best.pth.tar)|
|&#9745; 13 June, 2022|[VGGNet<sup>2](model/VGGNet.py)|30 / 30m-35s |0.9135|[VGGNet-best.pth.tar](model/VGGNet_Pretrained/best.pth.tar)|
|&#9744;              |[InceptionNet]()||
|&#9744;              |[ResNet]()||
### Notes
- Device on **GPU:** NVIDIA GeForce GTX 1070, **CPU:** Intel i7-7700K, **RAM:** 32GB and Win10 **System**.
- Training loss curves can be seen in [**Loss Curves**](model/Loss Curves) folder.
- <sup>1</sup> Considering the image size of Fashion-MNIST, here in AlexNet has some tiny differences with the original AlexNet Framework, and we don't take the seperate group and LRN structure as well.
- <sup>2</sup> Considering the image size of Fashion-MNIST, here in VGGNet16, we delete the last two block of convolutional layers, which are layer eight to thirteen.

## References
1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.
2. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. Advances in neural information processing systems, 25.
3. Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.
