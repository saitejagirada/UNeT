# Urban Scene Semantic Segmentation for Autonomous Perception

##  Project Overview
This repository contains a PyTorch implementation of a Semantic Segmentation pipeline designed for urban driving environments. Using a custom U-Net architecture, the model performs dense, pixel-wise classification across 20 distinct object classes (e.g., drivable surfaces, vehicles, pedestrians, infrastructure) based on the Cityscapes dataset.

The core focus of this project was overcoming severe data ingestion bottlenecks—specifically, engineering a robust data pipeline to dynamically clean and map noisy, compression-artifacted RGB ground-truth masks into precise class-index tensors for multi-class cross-entropy optimization.

##  Key Engineering Achievements
* **Dynamic Ground-Truth Sanitization:** The original dataset masks suffered from severe RGB compression artifacts, causing out-of-bounds index errors during loss calculation. I engineered a custom `Dataset` class utilizing a vectorized Euclidean distance color-mapper to map noisy RGB pixels to their nearest valid class centroid in the Cityscapes palette on the fly.
* **CUDA Exception Mitigation:** Resolved critical GPU device-side asserts (memory poisoning) by correctly masking void/unlabeled pixels (`ignore_index=19`) and ensuring dimensional accuracy across the forward pass.
* **Compute-Optimized Training:** Successfully trained the fully convolutional network (FCN) under strict memory constraints. The model demonstrates robust spatial understanding of macro-structures (roads, sidewalks, vehicles) even under aggressive input downsampling (256x96).

##  Qualitative Results
*(Insert your original and mask images here)*
The model effectively partitions the scene geometry. Drivable surfaces (purple) are cleanly separated from sidewalks (pink), and dynamic obstacles like vehicles (red) are distinctly localized despite the low spatial resolution.

##  Limitations & Areas for Improvement (Where the model lags)
While the model captures the global scene context excellently, there are known limitations based on the current architecture and compute constraints:
1. **Attenuation of High-Frequency Details:** Due to the aggressive spatial downsampling (256x96) required for training on limited compute, thin and distant objects (traffic poles, distant pedestrians, traffic lights) are swallowed by the Max Pooling layers in the encoder. 
2. **Lack of Multi-Scale Context:** The standard U-Net relies on basic skip-connections. For complex urban scenes, integrating an Atrous Spatial Pyramid Pooling (ASPP) module (as seen in DeepLabV3) would dramatically improve the model's ability to segment objects at varying scales.
3. **Class Imbalance:** Pixels representing roads and skies heavily outnumber pixels representing pedestrians or bicycles. Future iterations will replace standard Cross-Entropy Loss with **Focal Loss** or use weighted loss multipliers to penalize the model heavier for missing minority classes.

##  Tech Stack
* **Framework:** PyTorch, Torchvision
* **Architecture:** U-Net (Fully Convolutional Network)
* **Data Processing:** NumPy, PIL (Pillow)
* **Optimization:** Adam Optimizer, Masked Cross-Entropy Loss
