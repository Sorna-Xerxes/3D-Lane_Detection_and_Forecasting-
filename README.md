# Lane Detection and Forecasting – An Enhanced 3D Spatiotemporal Convolutional Network for Intelligent Lane Navigation in ADAS & AD 
![image](https://github.com/user-attachments/assets/1b704d56-c9cf-424e-8e56-99f5b54bc97c)

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
### Software Requirements:
```bash

```
### Hardware Requirements:
- The NVIDIA GeForce GTX 1650 in DELL G3 laptop for initial experiments
- The best-performing Network was executed on the University’s High-Performance Computing (HPC) system for up to 162 epochs.


### STM32 Cube IDE configurtions:
![image](https://github.com/user-attachments/assets/1edb2786-5adc-478e-a4d0-8a8df31c877e)

![Task 1 printout](https://github.com/user-attachments/assets/27d47c7b-5fde-479e-8fb3-9a83fcd615ce)

### VSCode Requirements:
![image](https://github.com/user-attachments/assets/fb8cde01-3c44-4cf6-85b8-4465bf757b2d)

### Python programming algorithm:
![python programming methodology_page-0001](https://github.com/user-attachments/assets/bd274363-8dd0-43e6-8fe4-92e57328f67f)

### Output of Task 1 & Task 2:
![Task1](https://github.com/user-attachments/assets/e20d1e41-4637-4e66-804e-3fe6f59beed7)

### Output of Task 3:
![image](https://github.com/user-attachments/assets/4c047949-f696-47fb-af1a-e12225b4ee47)

## DEPENDENCIES

**IMU Headers:** Declarations and prototypes for an accelerometer module, serving as an interface between application code and accelerometer functionality.

Declarations specific to the LSM303AGR 3D accelerometer and magnetometer, including register addresses and function prototypes.

Declarations for the STM32F3 Discovery board's accelerometer, providing macros and function prototypes

**IMU Source Files:** Implementation of functions declared in lsm303agr.h, handling interactions with the LSM 303AGR sensor.

Implementation of functions from stm32f3 discovery accelerometer.h, tailored for the STM32F3 Discovery board's accelerometer.

**Driver Dependency:** Dependency on the L3G20 gyroscope sensor, possibly including header files, source files, or precompiled libraries for additional motion data or functionality.

**MonoSLAM (Monocular Simultaneous Localization and Mapping) package:** It enables real-time localization and mapping using a single camera

**VIO:** It improve a robot's understanding of its surroundings by combining visual and inertial data

## HOW TO RUN (Terminal Commands)
```bash
roscd mono-slam/conf
rosrun mono-slam mono-slam conf.cfg /camera/image_raw:=/csi_cam_0/image_raw

sudo apt update -y
sudo apt-get install -y libconfig++-dev
roslaunch jetson_csi_cam jetson_csi_cam.launch

roslaunch mono-slam start_rviz.launch
roslaunch mono-slam static_tranform_world.launch 
```

## TABLE OF CONTENTS

| File Name        | Brief           |
| ------------- |:-------------:|
| ES-Project2-Task1.zip      | Contains the entire STM32 firmware that is to be flashed onto the STM32 for Task1 |
| ES-Project2-Task2.zip      | Contains the entire STM32 firmware that is to be flashed onto the STM32 for Task2|
| mono-slam.zip      | contains the slam Cmakes for ROS      |
|  my_jetson_robot.zip      | contains the IMU for ROS      |

## CREDITS
The STM32F3 discovery board and Jetson Nano were provided as part of the project materials for the completion of our coursework in Embedded Systems for the MSc in Robotics, AI, and Autonomous Systems at City, University of London.
