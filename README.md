# ECC2024: Reinforced Model Predictive Control via Trust-Region Improved Quasi-Newton Policy Optimization
Accompanying repository for our work "Reinforced Model Predictive Control via Trust-Region Improved Quasi-Newton Policy Optimization".

## Abstract
Model predictive control can control nonlinear systems under consideration of constraints optimally. The control performance depends on the model accuracy and the prediction horizon. Recent advances propose to use reinforcement learning applied to a parameterized model predictive controller to recover the optimal control performance even if an imperfect model or short prediction horizons are used. However, common reinforcement learning algorithms rely on first order updates, which only have a linear convergence rate and hence need an excessive amount of dynamic data, which is difficult to obtain for real-world applications. More elaborate algorithms like Quasi-Newton updates are intractable if common function approximators like neural networks are used due to the large number of parameters.

In this work, we exploit the small amount of parameters, which typically arise when using parameterized model predictive controllers, in a trust-region constrained Quasi-Newton training algorithm for deterministic policy optimization in reinforcement learning with a superlinear convergence rate. We show that the second order sensitivity tensor of the optimal solution of the model predictive controller, which is required in the deterministic policy Hessian, can be calculated by the solution of a linear system of equations. We apply the proposed algorithm to a case study and illustrate how it improves the data efficiency and accuracy compared to other algorithms.

## About this repository
Dear reader,
welcome to this repository. You'll find here the code that was created to produce the results for our work "Reinforced Model Predictive Control via Trust-Region Improved Quasi-Newton Policy Optimization". Please don't hesitate to write us a message, if you have any questions. To better understand the structure of this repository and our code please read the overview below.

## Instructions
All our results are created using Python code (using Jupyter Notebooks).
To run the code please follow these steps:
1) Install the anaconda environment ``.environment.yml`` via 
   ```
   conda env create -f PATH/TO/.environment.yml
   ```
2) Activate the environment
   ```
   conda activate ECC24_TR_Improved_QN_PO_for_MPC_in_RL
   ```
3) Clone [this repository](https://github.com/DeanBrandner/Reinforced_MPC.git) from GitHub and checkout the correct version
    ```
    git clone https://github.com/DeanBrandner/Reinforced_MPC.git PATH/TO/Reinforced_MPC 
	cd PATH/TO/Reinforced_MPC
	git checkout tags/v0.0.1
	```
5) Add the package to the ``conda`` environment
   ```
   conda develop PATH/TO/Reinforced_MPC
   ```

The results from Figure 1 can be reproduced by first running the following four files in the folder ``Agent_Comparison``
* ``training_GD_MPC.ipynb``
* ``training_TRGD_MPC.ipynb``
* ``training_QN_MPC.ipynb``
* ``training_TRQN_MPC.ipynb``

The plot is then generated by executing ``plot_CLC.ipynb``

Table II can be reproduced by first generating the test set via ``generate_test_data.py`` and then evaluating all methods with ``evaluate_test_data.py``, which saves the results in a ``*.xlsx`` file.
