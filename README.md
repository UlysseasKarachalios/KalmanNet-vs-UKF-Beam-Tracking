# KalmanNet-vs-UKF-Beam-Tracking
Solution of the Beam Tracking problem using KalmanNet network for my BSc Thesis "Model-Based and Data-Driven Kalman Filtering"
## Beam Tracking problem
We deal with the Beam/User Tracking Problem as formulated in paper "Beam Tracking for Distributed Millimeter-Wave Massive MIMO Systems
Based on the Unscented Kalman Filter"[Paper 1]
* Available at : [ieeeXplore](https://ieeexplore.ieee.org/document/9672140)
## KalmanNet code
The solution uses the KalmanNet implementation from paper "KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics" 
* Available at : [arxiv](https://arxiv.org/abs/2107.10043)  ,  [ieeeXplore](https://ieeexplore.ieee.org/document/9733186).
* Github repo : https://github.com/KalmanNet/KalmanNet_TSP
## Unscented Kalman filter code
We also use the Unscented Kalman Filter implementation from FilterPy Library
* Documentation: https://filterpy.readthedocs.io/en/latest/#
* Github repo : https://github.com/rlabbe/filterpy
## Requirements 
Check requirements.txt to install essential libraries
## Code execution
* main_BTD_Thesis.py : (inside the KalmanNet-vs-UKF-Beam-Tracking directory ) python main_BTD_Thesis.py or python ./main_BTD_Thesis.py
* main_BTD_Thesis_prompt_based.py : (inside the KalmanNet-vs-UKF-Beam-Tracking directory ) main_BTD_Thesis_prompt_based.py or python ./main_BTD_Thesis_prompt_based.py
## Main types 
* main_BTD_Thesis.py : Execution of the thesis case studies in Subsections 4.8.1-4.8.3 (Case1 - Case3) with hardcoded parameters. Already hardcoded in the first subcase of Case3 ((sigma_r)^2 = 10e-4 ).
* main_BTD_Thesis_prompt_based.py : Execution of the thesis case studies in Subsections 4.8.1-4.8.3 (Case1 - Case3) arbitrarily via prompt insertion.
## Case study data 
Case study data can be found in this [subdirectory](https://github.com/UlysseasKarachalios/KalmanNet-vs-UKF-Beam-Tracking/tree/main/Simulations/Beam_Tracking_Distributed)
## Types of state-space models in Beam-Tracking-Distributed subdirectory
* BTD_parameters_Realistic.py : State space model using sampling/time interval D_t = 0.0001(Time interval used in [Paper1](https://github.com/UlysseasKarachalios/KalmanNet-vs-UKF-Beam-Tracking/tree/main?tab=readme-ov-file#beam-tracking-problem) )
* BTD_parameters_Realistic_2.py : State space model using sampling/time interval D_t = 0.01
