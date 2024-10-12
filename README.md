# Adaptive Dynamic Fusion of Multi-Modality Features for Enhanced Image Representation
We propose a novel model based on Vision Transformers and an adaptive feature fusion network. Our model comprises a multi-level feature decoupling layer that decouples global and modality-specific features, followed by an attention-based adaptive dynamic feature fusion strategy.Our approach not only preserves information from source images but also produces fused images with high contrast and clear texture details.
## Project Structure

```
|-- ADF-Fusion of multiple modalities images
    |-- data
        |   |-- data_processed.h5
    |-- Train_dataset
        |   |-- ir
        |   |-- vi
    |-- Models
        |   |-- Model.pth
    |-- utils
        |   |-- Evalustor.py
        |   |-- dataprocessing.py
        |   |-- dataset.py
        |   |-- img_read_save.py
        |   |-- loss.py
    |-- Neural_network.py
    |-- README.md
    |-- requirements.txt
    |-- train.py
    |-- test.py
```
## Requirements
einops==0.4.1
kornia==0.2.0
numpy==1.21.5
opencv_python==4.8.1.78
scikit_image==0.19.2
scikit_learn==1.0.2
scipy==1.7.3
tensorboardX==2.5.1
timm==0.4.12
torch==1.12.1+cu113
torchvision==0.13.1+cu113
```
pip install -r requirements.txt
```
## Explanatory Note 
This project has made improvements to CDDFuse(https://github.com/Zhaozixiang1228/MMIF-CDDFuse.git) and achieved significant enhancements in some experimental metrics through thorough experimentation.As the article is currently in the submission stage, we are releasing only a portion of the code and the complete dataset. Upon acceptance for publication, we will make all of the code and trained model files publicly available.
