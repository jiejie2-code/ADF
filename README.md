# Adaptive Dynamic Fusion of Multi-Modality Features for Enhanced Image Representation
We propose a novel model based on Vision Transformers and an adaptive feature fusion network. Our model comprises a multi-level feature decoupling layer that decouples global and modality-specific features, followed by an attention-based adaptive dynamic feature fusion strategy.Our approach not only preserves information from source images but also produces fused images with high contrast and clear texture details.
## Project Structure

```
|-- Multimodal-Sentiment-Analysis
    |-- Config.py
    |-- main.py
    |-- README.md
    |-- requirements.txt
    |-- Trainer.py
    |-- data
    |   |-- .DS_Store
    |   |-- test.json
    |   |-- test_without_label.txt
    |   |-- train.json
    |   |-- train.txt
    |   |-- data
    |-- Models
    |   |-- CMACModel.py
    |   |-- HSTECModel.py
    |   |-- NaiveCatModel.py
    |   |-- NaiveCombineModel.py
    |   |-- OTEModel.py
    |   |-- __init__.py
    |-- src
    |   |-- CrossModalityAttentionCombineModel.png
    |   |-- HiddenStateTransformerEncoderCombineModel.png
    |   |-- OutputTransformerEncoderModel.png
    |-- utils
        |-- common.py
        |-- DataProcess.py
        |-- __init__.py
        |-- APIs
        |   |-- APIDataset.py
        |   |-- APIDecode.py
        |   |-- APIEncode.py
        |   |-- APIMetric.py
        |   |-- __init__.py
```
