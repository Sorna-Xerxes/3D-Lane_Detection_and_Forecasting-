# Lane Detection and Forecasting – An Enhanced 3D Spatiotemporal Convolutional Network for Intelligent Lane Navigation in ADAS & AD 

## ABSTRACT
Lane line detection has gained much recognition in the past years as a basic ingredient for both Advanced Driving Assistance Systems (ADAS) and Autonomous Driving (AD). It is very crucial for transport safety and navigation because it helps the systems take actions against driving out of lane boundaries and makes safe manoeuvres. However, conventional 2D models cannot extract temporal dependencies effectively and often fail to work well in dynamic environments. While some of the existing 3D approaches are indeed effective in feature extraction along both the spatial and temporal dimensions, their execution is fraught with high computational costs and complexities. 

This paper proposes an improved model that presents a balance between spatial and temporal information for effective lane detection. Two networks were developed in this study, both utilizing a 3D ResNet backbone encoder combined with the PINet decoder. The first network was enhanced with a Neck—incorporating a Self-Attention layer and Feature Pyramid Network. The second network was equipped with a detection Head Region of Interest for precise lane prediction and segmentation. The proposed second model achieved an improved accuracy of 93.33%, demonstrating significant reductions in false negatives, thereby outperforming the previous 3D technique while considerably reducing computational requirements.

Further work is warranted to integrate real-time constraints while exploring lightweight architectures for computational efficiency optimization, further enhancing performance and reliability under diverse real-world conditions.

## METHODOLOGY
### Proposed Network 1
![image](https://github.com/user-attachments/assets/9cbaf50a-e55f-4737-9c67-a8cd19a1a444)

### Proposed Network 2
![image](https://github.com/user-attachments/assets/ce53174e-5d51-402e-a098-d71cd70c89c8)

## REQUIREMENTS
### Dataset
| Attribute               | Details                                      |
|-------------------------|----------------------------------------------|
| **No. of Training Clips** | 3,626                                       |
| **No. of Testing Clips**  | 2,782                                       |
| **Frames per Clip**       | 20 frames                                   |
| **Image Resolution**      | 1280 × 720 pixels                           |
| **Annotation Format**     | Discrete lane points in JSON format         |
| **Driving Scenario**      | Highway scenes with multiple lanes          |
| **Conditions**            | Varying lighting, weather, and road environments |

### Software Dependencies
```bash
import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
import datetime as dt
import time
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import networkx as nx
```
### Hardware
- The NVIDIA GeForce GTX 1650 in DELL G3 laptop for initial experiments
- The best-performing Network was executed on the University’s High-Performance Computing (HPC) system for up to 162 epochs.
## PRE-PROCESSING
![image](https://github.com/user-attachments/assets/fee65124-2b35-4579-a70e-28549b3f6063)

## EXPERIMENTS
### Network 1
![image](https://github.com/user-attachments/assets/7d59dcb8-3c3d-4714-b58c-be87c9a7cbcf)
### Network 2
![image](https://github.com/user-attachments/assets/b8123147-39ad-41e4-9811-97a348c55122)

## RESULT (High Performance Computing)
![image](https://github.com/user-attachments/assets/b6ed97cc-e4a3-4054-b3dc-f6f5d435ac6e)
| Method                                       | Dimension | Accuracy % | FP     | FN     |
|----------------------------------------------|-----------|------------|--------|--------|
| **Line-CNN**                                 | 2D        | 96.87%     | 0.0442 | 0.0197 |
| **CLRNet**                                   | 2D        | 96.83%     | 0.0237 | 0.0238 |
| **SCNN**                                     | 2D        | 96.53%     | 0.0617 | 0.0180 |
| **PINet**                                    | 2D        | 93.36%     | 0.0467 | 0.0254 |
| **PolyLaneNet**                              | 2D        | 93.36%     | 0.0942 | 0.0933 |
| **My (PINet_3DResNet_ROI_Focal_LineIOU)**    | 3D        | 93.33%     | 0.0515 | 0.0822 |
| **Previous Work (PINet_3DResNet50)**         | 3D        | 91.34%     | 0.1138 | 0.1101 |

## FUTURE WORKS
- **Development of Hybrid Encoder-Decoder Architectures:**  Can be refined by integrating hybrid Transformer-based modules
- **Adaptive Feature Aggregation Strategies:** Integrating adaptive feature aggregation techniques or even including ROI after FPN may perform better
- **Incorporation of Dilated Convolutions:** At specific stages to capture a broader spatial context without significantly increasing the computational load

## CONTRIBUTIONS
Please open an issue to discuss your ideas, or submit a pull request if you've implemented a feature or bug fix.

## LICENSE
This project is licensed under the MIT License.
