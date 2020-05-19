# CSI-ComSensing-AE

This the Tensorflow implementation of paper "Acquiring Measurement Matrices via Model-Based Deep Learning for Sparse Channel Estimation in mmWave Massive MIMO Systems"


## Prerequisites
Here is the software environment we used
1. Python 3.6 
2. Tensorflow 1.12
3. Gurobi 8.0.1
4. Matlab R2017a

## Get Started

To reproduce our results, please follow the three steps:

1. Prepare dataset following the details in the below section "Dataset Preparation";  

2. Run the script "channel_main_call.sh" in terminal;

3. Read the results by running "readres.py" in the folder utils;

### Dataset Preparation
You can prepare your own dataset by following :

1. Generate the spatial channels following the instruction on [DeepMIMO](https://www.deepmimo.net/) and using the parameters given in the paper "Acquiring Measurement Matrices via Model-Based Deep Learning for Sparse Channel Estimation in mmWave Massive MIMO Systems". Also, you can skip this step 1 and directly download our used dataset from [this link](https://drive.google.com/file/d/1Ccwh8XdW3AXNMQ62j6D5Ndd4qRVxTbja/view?usp=sharing).

2. Access the generated spatial channel vectors and transform to beamspace channels by running the Matlab Script "./dataset/beamspace_channels.m".

## Built With

* [DeepMIMO](https://www.deepmimo.net/) - for Dataset Generation

## Acknowledgments

* We have adopted a part code from [L1AE](https://github.com/wushanshan/L1AE) for model construction.
