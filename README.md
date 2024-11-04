# Multi-Sensory Fusion for Autonomous Camera Control and Navigation
![image](https://github.com/user-attachments/assets/1b704d56-c9cf-424e-8e56-99f5b54bc97c)

## ABSTRACT
**Task 1,** a Python script leveraging computer vision techniques and OpenCV library will be developed to detect humans within a camera's view. A control loop will then adjust the camera orientation to keep the human centrally located in the frame.

**Task 2** involves writing firmware for an STM32 microcontroller board to collect IMU data and correct any camera misalignment caused by robot movement or environmental factors. Sensor fusion algorithms will be implemented to enhance tracking accuracy.

**Task 3** focuses on integrating a Visual-Inertial Odometry (VIO) system to improve the robot's spatial awareness and movement accuracy. This involves fusing visual information from cameras with inertial data from IMUs, enabling precise navigation and mapping without GPS.

The project outcomes include gaining proficiency in implementing computer vision algorithms, developing real-time firmware for microcontrollers, applying sensor fusion algorithms, and understanding Visual-Inertial Odometry principles. Soft skills such as problem-solving, project management, teamwork, and communication will also be enhanced throughout the project.

Overall, this project seeks to advance the capabilities of autonomous systems in dynamic environments through the integration of multiple sensory inputs for robust navigation and control.

## SYSTEM OVERVIEW
The following **block diagram** illustrates the system architecture and data flow between the components:

![image](https://github.com/user-attachments/assets/b70b8875-493a-4d55-9569-f327590ae4d2)

### Hardware Requirements:
- STM32 F3 Discovery
- Jetson Nano
- Camera - Raspberry Pi
- Servo Motors x 2 (180°)
- Pan-Tilt Camera Mount
- Ethernet Cable
- Power Supply Adapter (5V)
- USB Cable

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
