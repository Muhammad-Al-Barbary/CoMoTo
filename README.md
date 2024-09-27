# CoMoTo: Unpaired Cross-Modal Lesion Distillation Improves Breast Lesion Detection in Tomosynthesis

###### Muhammad Alberb, Marawan Elbatel, Aya Elgebaly, Ricardo Montoya-del-Angel, Xiaomeng Li, Robert Martí

This is the official implementation of the framework CoMoTo: Unpaired **C**r**o**ss-**Mo**dal Lesion Distillation Improves Breast Lesion Detection in **To**mosynthesis

## Abstract
Digital Breast Tomosynthesis (DBT) is an advanced breast imaging modality that offers superior lesion detection accuracy compared to conventional mammography, albeit at the trade-off of longer reading time. Accelerating lesion detection from DBT using deep learning is hindered by limited data availability and huge annotation costs. A possible solution to this issue could be to leverage the information provided by a more widely available modality, such as mammography, to enhance DBT lesion detection. In this paper, we present a novel framework, CoMoTo, for improving lesion detection in DBT. Our framework leverages unpaired mammography data to enhance the training of a DBT model, improving practicality by eliminating the need for mammography during inference. Specifically, we propose two novel components, Lesionspecific Knowledge Distillation (LsKD) and Intra-modal Point Alignment (ImPA). LsKD selectively distills lesion features from a mammography teacher model to a DBT student model, disregarding background features. ImPA further enriches LsKD by ensuring the alignment of lesion features within the teacher before distilling knowledge to the student. Our comprehensive evaluation shows that CoMoTo is superior to traditional pretraining and image-level KD, improving performance by 7% Mean Sensitivity under low-data setting.

![CoMoTo_overview](https://github.com/user-attachments/assets/b0af0e0d-8f02-4067-8bae-7bf505fe3946)


## Getting Started
```
pip install -r requirements.txt
```

## Running Framework
1. add a dataset function to comoto/data/datasets.py
2. add a configuration json file to comoto/configs
3. run ```python main.py --config_name configs --mammo``` or ```python main.py --config_name configs --dbt``` to run the training and evaluation code for mammography or DBT

## Results
Our framework improves performance over alternative methods, especially in low-data setting
<img width="871" alt="Screenshot 2024-09-24 200959" src="https://github.com/user-attachments/assets/9e8c56e8-a6fd-4a1d-a08c-e17086f7da63">


### Acknowledgment
Our code is built on [MONAI](https://monai.io/core.html).

### Citation
````
@article{alberb2024CoMoTo,
  title={CoMoTo: Unpaired Cross-Modal Lesion Distillation Improves Breast Lesion Detection in Tomosynthesis},
  author={Muhammad Alberb and Marawan Elbatel and Aya Elgebaly and Ricardo Montoya-del-Angel and Xiaomeng Li and Robert Martí},
  journal={arXiv preprint arXiv:2407.17620},
  year={2024}
}
````



