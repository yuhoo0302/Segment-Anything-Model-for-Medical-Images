# Segment-Anything-Model-for-Medical-Images
Implementation of "Segment Anything Model for Medical Images?" in pytorch.
Arxiv link:[https://arxiv.org/pdf/2304.14660.pdf](https://arxiv.org/pdf/2304.14660.pdf)

## Usage

### 1. Prepare your data in the following format and prepare the segmentation information _json_ files corresponding to each test set.

 <details>  
  
 <summary>Json file content.</summary>
  
 Info refers to the segmentation target in this dataset, while color is the ground truth pixel value corresponding to the target.
  
 ```
 {
    "info": {
        "1": "LeftVentricle",
        "2": "LeftVentricularMyocardium",
        "3": "RightVentricle"
    },
    "color": {
        "1": 85,
        "2": 170,
        "3": 255
    }
 }
 ```
</details>

<details>  
  
 <summary>Dataset distribution.</summary>

 ```
  train: ../data/train_data/images/
  val: ../data/train_data/images/
  test: ../data/test_data/dataset_name/images/
  
  ├── train_data          
  │   ├── images        
  │   │   ├── 000001.png
  │   │   ├── 000002.png
  │   │   └── 000003.png
  │   └── labels         
  │       ├── 00001.png
  │       ├── 00002.png
  │       └── 00003.png
  └── val_data           
  |   ├── images        
  |   │   ├── 000001.png
  |   │   ├── 000002.png
  |   │   └── 000003.png
  |   └── labels         
  |       ├── 00001.png
  |       ├── 00002.png
  |       └── 00003.png
  └── test_data          
      ├── dataset1        
      │   ├── images
      |   |     ├── 000001.png
      |   │     ├── 000002.png
      |   |     └── 000003.png
      │   └── labels
      |         ├── 000001.png
      |         ├── 000002.png
      |         └── 000003.png
      └── dataset2         
          ├── images
          |     ├── 000001.png
          │     ├── 000002.png
          |     └── 000003.png
          └── labels
                ├── 000001.png
                ├── 000002.png
                └── 000003.png
  ```
</details>

### 2. Generate embedding for each single image.

```
$ python pre_grey_rgb2D.py         #for ViT-B model
$ python pre_grey_rgb2D_Huge.py    #for ViT-H model
```

### 3. Finetune SAM with your own data.

```
$ python train_only_box.py    
```

### 4. Test on finetuned checkpoints.

```
$ python test_only_box.py    
```

### 5. Calculate your metric.

```
$ python cal_matric.py       
```
