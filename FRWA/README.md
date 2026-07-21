FRWA certificate code files for reference and execution, without incorporating the nonlinear safe velocity constraint (instead opting for a flat 0.5 m/s absolute value velocity limit on each velocity component). We outline files and a brief description of included functions. We additionally present how to run code in order to execute our implementation.

For running code: Run run_cluster_exp_mask.py. Choose to modify lines 9 and 10 in order to change the positional limit (presented as absolute value). In order to change the initial controller file path, line 20 can be modified. Note that modifying the controller itself should only be done such that the architecture is consistent to the original "fixed_controller_20n_manhattan.pt" file.

run_cluster_exp_mask.py: File which executes code to learn a verified FRWA certificate and controller after various CEGIS iterations, given an initial controller file (in PyTorch format) trained using code in https://github.com/act3-ace/SafeRL in addition to the "design-for-verification" scheme.

training_exp_mask.py: File containing functions and classes for training the FRWA certificate and controller.

queries_mask.py: File containing verification functions using Marabou for run_cluster_exp_mask.py.

generate_combined_model_torch.py: File containing helper functions for creating a singular ONNX network composed of relevant networks to provide to Marabou. Functions here are used in run_cluster_exp_mask.py.

attempt_conversion.py: File containing helper class for training_exp_mask.py, to ensure controller network architecture consistency.

convertsinglenetwork.py: File containing helper function to convert a singular PyTorch network to an ONNX network to provide to Marabou. Function here is used in run_cluster_exp_mask.py.