
## Data Preparation
1. Download the training dataset of [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) and [VinBigData Chest X-ray Abnormalities Detection](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/data) challenge. Note that we only use their training set as labels of testing set are not available. 
2. Use `data/preprocess.py` to preprocess the two datasets respectively. The output files should be `*.png`.
3. Move the repartition files `rsna_data.json` and `vin_data.json` to corresponding data roots and rename to `data.json`.

The final structure of datasets should be as following:
```python
├─DATA_PATH
│ ├─rsna-pneumonia-detection-challenge   # data root of RSNA dataset
│ │ ├─train_png_512   # preprocessed images of rsna dataset 
│ │ │ ├─xxx.png
│ │ │ ├─ ......
│ │ ├─data.json   # repartition file of rsna dataset (renamed from "rsna_data.json")
│ ├─VinCXR   # data root of VinBigData dataset
│ │ ├─train_png_512   # preprocessed images of VinBigData dataset
│ │ │ ├─xxx.png
│ │ │ ├─ ......
│ │ ├─data.json   # repartition file of VinBigData dataset (renamed from "vin_data.json")
```

The `data.json` is a dictionary that storing the data repartition information:

```json
{
  "train": {
    "0": ["*.png", ], // The known normal images for one-class training
    "unlabeled": {
          "0": ["*.png", ], // normal images used to build the unlabeled dataset
    	  "1": ["*.png", ]  // abnormal images used to build the unlabeled dataset
    }
  },
  
  "test": {
  	"0": ["*.png", ],  // normal testing images
  	"1": ["*.png", ]  // abnormal testing images
  }
}
```
## Model Preparation
Download pre-trained backbones [simCLRV2](https://drive.google.com/file/d/1X0mNsmZKLnkPTlk6Ji-HxC27JO1JZylS/view?usp=drive_link) and [Barlow](https://drive.google.com/file/d/1tuLpVD0dfgn15hJ4gZCp9dX9_OZ3nAur/view?usp=drive_link).

**Train**  

Train the proposed network with simCLRv2 as the backbone on VinBigData with different inlier and outlier
```
python siamese_main.py --config cfgs/Vin_siamese.yaml --data_ratio  0.5  --outlier_ratio 0;
```
Train the proposed network with barlow as the backbone on VinBigData with different inlier and outlier
```
python siamese_main.py --config cfgs/Vin_barlow.yaml --data_ratio  0.5  --outlier_ratio 0;
```

**Evaluation**
```
python siamese_main.py --config cfgs/Vin_siamese.yaml --k 1 --mode test  --geometric_mean  --data_ratio 0.5  --normalization  --epoch 99
```
## Citation

```
@inproceedings{Liu2024knn,
title = {Deep Nearest Neighbors for Anomaly Detection in Chest X-Rays},
author = {Liu, Xixi and Alv{\'e}n, Jennifer
and H{\"a}ggstr{\"o}m, Ida
and and Christopher, Zach},
booktitle = {Machine Learning in Medical Imaging},
year = {2024}
publisher={Springer Nature Switzerland},
}
```

## Acknowledgement

Our data preparartion code is adapted from [DDAD](https://github.com/caiyu6666/DDAD), thanks a lot for their great work!

 

 
