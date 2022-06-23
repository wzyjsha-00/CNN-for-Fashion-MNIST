# CNN Implementation for Fashion-MNIST in PyTorch

## Introduction  
This repository is the implementation of some classical **C**onvolutional **N**eural **N**etworks for [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) in PyTorch.
Please feel free to ask anything!
- [**LeNet**](model/LeNet.py), [*Paper Link*](https://ieeexplore.ieee.org/document/726791)
- [**AlexNet**](model/AlexNet.py), [*Paper Link*](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf)
- [**VGGNet**](model/VGGNet.py), [*Paper Link*](https://arxiv.org/abs/1409.1556)
- [**InceptionNet**](model/InceptionNet.py), [*Paper Link*](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Szegedy_Going_Deeper_With_2015_CVPR_paper.html)
- [**ResNet**](model/ResNet.py), [*Paper Link*](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)

## Baseline Environment and Setup
```python  
python 3.9.11
cuda 11.0.3
cudnn 8.0.5 
torch 1.7.1
```
- **Install pytorch:**  ```pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html```
- **Install dependencies:** ```pip install -r requirements.txt```
- **Download** the [Fashion-MNIST](https://1drv.ms/u/s!AjRAq1taDd8ygo19T6XNf6S5T9HT5Q) then put the unzipped file into the [data](data) folder
- **Download** the **best.pth.tar** from corresponding pre-trained model parameter folder (seeing the table in **Working Record**) and put the **best.pth.tar** file into the [model](model) folder
- **Choose** the model you need in Line125-129 of [train.py ](train.py) and **Check** the [hyper parameters](config/hyper_param.py)

## Project Structure
```buildoutcfg
.
|-- config/             
|-- data/
|-- model/
|-- resources/
|-- train.py
|-- requirements.txt
|-- .gitignore
|-- LICENSE
|-- README.md
```


##  Working Record
|        State        |Network|Train Epochs / Time|Best Test Acc|Model Param|
|:-------------------:|:-----:|:-----------------:|:-----------:|:---------:|
|&#9745; 09 June, 2022|[LeNet](model/LeNet.py)|30 / 18m-29s|0.9016|[LeNet-best.pth.tar](https://1drv.ms/u/s!AjRAq1taDd8ygoxyiPfXntiHJUpnyg?e=RmTujU)|
|&#9745; 11 June, 2022|[AlexNet<sup>1](model/AlexNet.py)|30 / 35m-39s|0.9219|[AlexNet-best.pth.tar](https://1drv.ms/u/s!AjRAq1taDd8ygox8xU7DnBkF8dsvjA?e=slHcKS)|
|&#9745; 13 June, 2022|[VGGNet<sup>2](model/VGGNet.py)|30 / 30m-35s |0.9135|[VGGNet-best.pth.tar](https://1drv.ms/u/s!AjRAq1taDd8ygo0AGbjJNEZyfuNnMQ?e=O7AOi9)|
|&#9745; 22 June, 2022|[InceptionNet](model/InceptionNet.py)|30 / 416m-54s|0.9274|[InceptionNet-best.pth.tar](https://1drv.ms/u/s!AjRAq1taDd8ygo120M9sOh9bJtrwQg?e=kewwJ7)|
|&#9745; 16 June, 2022|[ResNet](model/ResNet.py)|30 / 82m-24s|0.9340|[ResNet-best.pth.tar](https://1drv.ms/u/s!AjRAq1taDd8ygox-sWyCXUrfu-bCOg?e=gchZpE)|
### Notes
- Device on **GPU:** NVIDIA GeForce GTX 1070, **CPU:** Intel i7-7700K, **RAM:** 32GB and Win10 **System**.
- Training loss curves can be seen in [**Loss Curves**](model/Loss_Curves) folder.
- <sup>1</sup> Considering the image size of Fashion-MNIST, here in AlexNet has some tiny differences with the original AlexNet Framework, and we don't take the seperate group and LRN structure as well.
- <sup>2</sup> Considering the image size of Fashion-MNIST, here in VGGNet16, we delete the last two block of convolutional layers, which are layer eight to thirteen.

## References
1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.
2. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. Advances in neural information processing systems, 25.
3. Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.
4. Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., ... & Rabinovich, A. (2015). Going deeper with convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1-9).
5. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).