# Learning Texture Invariant Representation for Domain Adaptation of Semantic Segmentation 

A pytorch implementation of [LTIR](https://arxiv.org/abs/2003.00867).

<img width="534" alt="image" src="https://user-images.githubusercontent.com/39029444/78094147-c9123800-740e-11ea-83b0-3ee28c2d305b.png">

### Requirements

* Python 3.6
* torch==1.2
* torchvision==0.4
* Pillow==6.1.0

### Stylizing dataset

We used code from [Style-swap](https://github.com/rtqichen/style-swap).

### Weight of Final Model

##### GTA5 to Cityscapes

* [ResNet](https://drive.google.com/file/d/1uwNFhrHYnTU-lAcs6hT4r_rg2Pqib-K1/view?usp=sharing)

* [VGG](https://drive.google.com/file/d/1gAjmwbg60JDIzE4oLxymr2Dwsco_xB5Q/view?usp=sharing)

##### SYNTHIA to Cityscapes

* [ResNet](https://drive.google.com/file/d/1q50tLjbzKZxOA-Wj_YWvs7bq85JtHTow/view?usp=sharing)

* [VGG](https://drive.google.com/file/d/1Gx4Pkav6XAWZQHlp5kzAPytb41zHSEtT/view?usp=sharing)

### Acknowledgement

This code is heavily borrowed from [AdaptSegNet](https://github.com/wasidennis/AdaptSegNet) and [BDL](https://github.com/liyunsheng13/BDL).
 
