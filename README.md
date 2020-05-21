# DeepBP-AE

This the Tensorflow implementation of paper "Acquiring Measurement Matrices via Deep Basis Persuit for Sparse Channel Estimation in mmWave Massive MIMO Systems"


## Prerequisites
Here is the software environment we used
1. Python 3.6 
2. Tensorflow 1.12
3. Gurobi 8.0.1
4. Matlab R2017a

## Get Started

To reproduce our results, just run the script "channel_main_call.sh" in terminal.

Or If you perfer to start from scatch, follow the steps below:

1. Prepare deepMIMO dataset. Details are given in the following section "Dataset Preparation";  

2. Train the autoencoder models and obtain the data-driven measurement matrix by running any python files named starting with 'main'. Please feel free to play around with changing the default parameters. 

3. Read the results by running "readres.py" in the folder utils;

### Dataset Preparation
You can prepare your own dataset by following :

1. Generate the spatial channels following the instruction on [DeepMIMO](https://www.deepmimo.net/). The parameters are given in the paper "Acquiring Measurement Matrices via Deep Basis Pursuit for Sparse Channel Estimation in mmWave Massive MIMO Systems". Alternatively, you can skip this step 1 and directly download our used dataset from [this link](https://drive.google.com/file/d/1Ccwh8XdW3AXNMQ62j6D5Ndd4qRVxTbja/view?usp=sharing).

2. Access all generated spatial channel vectors and save as a MAT-file. This can be done by running the Matlab Script "./datasets/access_channels.m".

3. Obtain the beamspace channel vectors and save as a MAT-file. by running the Matlab Script "./datasets/beamspace_channels.m". 

## Built With

* [DeepMIMO](https://www.deepmimo.net/) - for Dataset Generation
* [L1AE](https://github.com/wushanshan/L1AE) - part code is adopted for model construction.
