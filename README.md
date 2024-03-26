# Segment-Anything-Model-for-Medical-Images

Implementation of "Segment Anything Model for Medical Images?" in pytorch --for finetuning the SAM with box prompts.

Arxiv link: [https://arxiv.org/pdf/2304.14660.pdf](https://arxiv.org/pdf/2304.14660.pdf)

MIA version link: [https://www.sciencedirect.com/science/article/pii/S1361841523003213](https://www.sciencedirect.com/science/article/pii/S1361841523003213)

#### Our work has been accepted by Medical Image Analysis (MedIA) 2023! 

## Usage

### 1. Prepare data and segmentation information _json_ files corresponding to each test set.

 <details>  
  
 <summary>Json file content.</summary>
  
 "Info" refers to the segmentation target in this dataset, while "color" is the ground truth pixel value corresponding to the target.
  
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
$ python pre_grey_rgb2D.py         --img_path  data/train_data/images    --gt_path data/train_data/labels                       --checkpoint sam_vit_b_01ec64.pth              #for preparing training data (embeddings) with ViT-B  
$ python pre_grey_rgb2D_Huge.py    --img_path  data/test_data            --gt_path data/test_data     --task_name 22_Heart       --checkpoint sam_vit_b_01ec64.pth              #for preparing testing data (embeddings) with ViT-B
```

### 3. Finetune SAM with your own data.

```
$ python train_only_box.py    --tr_npz_path data/precompute_vit_b/train  --val_npz_path data/precompute_vit_b/valid --model_type vit_b # finetune ViT-B
                              --tr_npz_path data/precompute_vit_h/train  --val_npz_path data/precompute_vit_h/valid --model_type vit_h # finetune ViT-H
```

### 4. Test on finetuned models and output the Dice results.

```
$ python test_only_box.py    
```

### 5. Calculation of all the indicators (Dice, IOU, HD, etc.).

```
$ python cal_matric.py       
```

# Our pretrained weights  
Checkpoints download path: [https://drive.google.com/drive/folders/1jry-07RxGYQnT9cQE8weuCZurDCu58pj?usp=sharing](https://drive.google.com/drive/folders/1jry-07RxGYQnT9cQE8weuCZurDCu58pj?usp=sharing)

# COSMOS 1050K Dataset

We collected and sorted 53 public datasets to build the large COSMOS 1050K medical image segmentation dataset.
Following are the links to the datasets used in our paper.

Ownership and license of the datasets belong to their corresponding original papers, authors, or competition organizers. If you use the datasets, please cite the corresponding paper or links.

<details>
<summary style="font-size:30px;">Links</summary>

### AbdomenCT-1K	
https://abdomenct-1k-fully-supervised-learning.grand-challenge.org/

### ACDC	
https://www.creatis.insa-lyon.fr/Challenge/acdc/

### AMOS 2022	
https://amos22.grand-challenge.org/

### AutoLaparo	
https://autolaparo.github.io/

### BrainPTM 2021	
https://brainptm-2021.grand-challenge.org/

### BraTS20	
https://www.med.upenn.edu/cbica/brats2020/data.html

### CAMUS	
https://www.creatis.insa-lyon.fr/Challenge/camus/databases.html

### CellSeg Challenge-NeurIPS 2022 	
https://neurips22-cellseg.grand-challenge.org/

### CHAOS	
https://zenodo.org/record/3431873#.YKIkTfkzbIU

### CHASE-DB1	
https://blogs.kingston.ac.uk/retinal/chasedb1/

### Chest CT Segmentation	
https://www.kaggle.com/datasets/polomarco/chest-ct-segmentation

### CRAG	
https://warwick.ac.uk/fac/cross_fac/tia/data/

https://warwick.ac.uk/fac/cross_fac/tia/data/mildnet/

### crossMoDA	
https://crossmoda.grand-challenge.org/

### CVC-ClinicDB	
https://polyp.grand-challenge.org/CVCClinicDB/

### DRIVE	
https://drive.grand-challenge.org/

### EndoTect 2020	
https://endotect.com/

### ETIS-Larib Polyp DB	
https://polyp.grand-challenge.org/

### FeTA	
https://feta.grand-challenge.org/

### HaN-Seg	
https://han-seg2023.grand-challenge.org/

### I2CVB	
https://i2cvb.github.io/

### iChallenge-AMD	
https://amd.grand-challenge.org/

### iChallenge-PALM	
https://palm.grand-challenge.org/

### IDRiD 2018	
https://idrid.grand-challenge.org/

### iSeg 2019	
https://iseg2019.web.unc.edu/

### ISIC 2018	
https://challenge.isic-archive.com/data#2016

### IXI	
https://brain-development.org/ixi-dataset/

### KiPA22	
https://kipa22.grand-challenge.org/

### KiTS19	
https://kits19.grand-challenge.org/

### KiTS21	
https://kits-challenge.org/kits21/

### Kvasir-Instrumen	
https://datasets.simula.no/kvasir-instrument/

### Kvasir-SEG	
https://datasets.simula.no/kvasir-seg/

### LiVScar	
https://figshare.com/articles/figure/Left_ventricular_LV_scar_dataset/4214622?file=6875637

### LUNA16
https://luna16.grand-challenge.org/

### M&Ms	
https://www.ub.edu/mnms/

### Montgomery County CXR Set	
https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/index.html

### MRSpineSeg	
https://www.spinesegmentation-challenge.com/

https://github.com/pangshumao/SpineParseNet

### MSD	
http://medicaldecathlon.com

### Multi-Atlas Labeling Beyond the Cranial Vault（Abdomen）：MALBCV-Abdomen	
https://www.synapse.org/#!Synapse:syn3193805/wiki/217752

### NCI-ISBI 2013	
https://wiki.cancerimagingarchive.net/display/Public/NCI-ISBI+2013+Challenge+-+Automated+Segmentation+of+Prostate+Structures

### PROMISE12	
https://promise12.grand-challenge.org/

### QUBIQ 2021	
https://qubiq21.grand-challenge.org/

### SIIM-ACR	
https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation

### SKI10	
https://ski10.grand-challenge.org/

### SLIVER07	
https://sliver07.grand-challenge.org/

### STARE	
https://cecas.clemson.edu/~ahoover/stare/

### TN-SCUI 2020	
https://tn-scui2020.grand-challenge.org/

### VerSe19&VerSe20	
https://github.com/anjany/verse

### Warwick-QU	
https://warwick.ac.uk/fac/cross_fac/tia/data/
https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest/download/

### WORD	
https://github.com/HiLab-git/WORD

### EPFL_EM
https://www.epfl.ch/labs/cvlab/data/data-em/

### ssTEM	
https://imagej.net/events/isbi-2012-segmentation-challenge

### TotalSegmentator	
https://github.com/wasserth/TotalSegmentator

### 4C2021 C04 TLS01	
https://aistudio.baidu.com/aistudio/projectdetail/1952488?channelType=0&channel=0
</details>

# Acknowledgments
Our code is based on [Segment Anything](https://github.com/facebookresearch/segment-anything) and [MedSAM](https://arxiv.org/abs/2304.12306). We appreciate the authors for their great works. We also sincerely appreciate all the challenge organizers and owners for providing the public medical image segmentation datasets.

# Citation
If you find the code useful for your research, please cite our paper.
```sh
@article{huang2024segment,
  title={Segment anything model for medical images?},
  author={Huang, Yuhao and Yang, Xin and Liu, Lian and Zhou, Han and Chang, Ao and Zhou, Xinrui and Chen, Rusi and Yu, Junxuan and Chen, Jiongquan and Chen, Chaoyu and others},
  journal={Medical Image Analysis},
  volume={92},
  pages={103061},
  year={2024},
  publisher={Elsevier}
}
