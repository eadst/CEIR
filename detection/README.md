# Detecting Text in Natural Image with Connectionist Text Proposal Network

## Requirements
* pytorch 1.0
* opencv3


## Data Preparation
```
img
│   1.jpg
│   2.jpg   
│		...
gt
│   gt_1.txt
│   gt_2.txt
|		...
```

## Train
1. config the `trainroot`,`testroot`in [config.py](config.py)
2. use fellow script to run
```sh
python3 train.py
```

## Test
[eval.py](eval.py) is used to test model on test dataset

1. config `model_path`, `data_path`, `gt_path`, `save_path` in [eval.py](eval.py)
2. use fellow script to test
```sh
python3 eval.py
```

## Predict 
[predict.py](predict.py) is used to inference on single image

1. config `model_path`, `img_path`, `gt_path`, `save_path` in [predict.py](predict.py)
2. use fellow script to predict
```sh
python3 predict.py
```

The project is still under development.

<h2 id="Performance">Performance</h2>

### [ICDAR 2015](http://rrc.cvc.uab.es/?ch=4)
only train on ICDAR2015 dataset with single NVIDIA 1080Ti

my implementation with my loss use adam and warm_up

| Method                   | Precision (%) | Recall (%) | F-measure (%) | FPS(1080Ti) |
|--------------------------|---------------|------------|---------------|-----|
| tbd  | tbd | tbd | tbd | tbd |