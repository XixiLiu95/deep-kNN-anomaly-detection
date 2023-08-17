
# Data Preparation
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

 

 