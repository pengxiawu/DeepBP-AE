# CSI-ComSensing-AE

This the Tensorflow implementation of paper ``Acquiring Measurement Matrices via Model-Based Deep Learning for Sparse Channel Estimation in mmWave Massive MIMO Systems"


## Prerequisites
Here is the software environment we used
1. Python 3.6 
2. Tensorflow 1.12
3. Gurobi 8.0.1
4. Matlab R2017a

## Get Started

To reproduce our results, please follow the three steps:
1. Prepare dataset following details in the below section "Dataset Preparation";  
2. Run the script "channel_main_call.sh" in terminal;
3. Read the results by running "readres.py" in the folder utils;

## Dataset Preparation
Following the instruction in 

## Built With

* [DeepMIMO](https://www.deepmimo.net/) - for Dataset Generation
* [L1AE](https://github.com/wushanshan/L1AE) - used a part code for model construction 

## Acknowledgments

* We have used some code from https://github.com/wushanshan/L1AE for model construction and https://www.deepmimo.net/ for Dataset Generation.
