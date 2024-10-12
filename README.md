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
