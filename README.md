# MLCM_MOE
all_predictions.csv contains the results of human-machine annotation; 
MoE.py is the code for implementing the human-machine integration of the original MoE framework (MLCM does not participate in the human-machine integration process); 
MoE_CM.py is the implementation code for our method; 
MoE_CM_Stacking.py is the implementation code for the ablation experiment where the gating network is changed to a stacking method; 
MoE_CM_thre.py is the implementation code for the ablation experiment where the thresholds set separately for each category are removed.

Due to a server failure, it is currently impossible to upload all the code. However, all the ideas of this paper are implemented in MoE_CM.py.
