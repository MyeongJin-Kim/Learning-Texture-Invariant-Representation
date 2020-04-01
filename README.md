# Learning Texture Invariant Representation for Domain Adaptation of Semantic Segmentation 

A pytorch implementation of [LTIR](https://arxiv.org/abs/2003.00867).

<img width="534" alt="image" src="https://user-images.githubusercontent.com/39029444/78094147-c9123800-740e-11ea-83b0-3ee28c2d305b.png">

### Requirements

* Python 3.6
* torch==1.2
* torchvision==0.4
* Pillow==6.1.0

### Preparing dataset

We used code from [Style-swap](https://github.com/rtqichen/style-swap) and [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

### Training

Initial weight

```
python train_gta2cityscapes.py --translated-data-dir /Path/to/translated/source --stylized-data-dir /Path/to/stylized/source
```

* [ResNet](https://drive.google.com/file/d/1aakvRd3OI8TBaBH7gfw1kgm4B0d7HZB1/view?usp=sharing)

* [VGG](https://drive.google.com/file/d/1fqJvVd1I65G2A_1GD9ZRIyuruTPRpnbq/view?usp=sharing)

### Evalutation

```
python evaluate_cityscapes.py --restore-from /Path/to/weight
python compute_iou.py /Path/to/Cityscapes/gtFine/val /Path/to/results
```

### Weight of Final Model

##### GTA5 to Cityscapes

* [ResNet](https://drive.google.com/file/d/1uwNFhrHYnTU-lAcs6hT4r_rg2Pqib-K1/view?usp=sharing)

* [VGG](https://drive.google.com/file/d/1gAjmwbg60JDIzE4oLxymr2Dwsco_xB5Q/view?usp=sharing)

##### SYNTHIA to Cityscapes

* [ResNet](https://drive.google.com/file/d/1q50tLjbzKZxOA-Wj_YWvs7bq85JtHTow/view?usp=sharing)

* [VGG](https://drive.google.com/file/d/1Gx4Pkav6XAWZQHlp5kzAPytb41zHSEtT/view?usp=sharing)

### Acknowledgement

This code is heavily borrowed from [AdaptSegNet](https://github.com/wasidennis/AdaptSegNet) and [BDL](https://github.com/liyunsheng13/BDL).
 
