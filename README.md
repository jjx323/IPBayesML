# IPBayesML: Programs for Inverse Problems of PDEs with Bayesian and Machine Learning Methods

## core: provide the basic components of coding statistical inverse problems
+ probability.py
+ noise.py
+ model.py
+ eigensystem.py
+ optimizer.py
+ sample.py
+ approximate_sample.py
+ linear_eq_solver.py
+ misc.py 

## SimpleSmooth
In this folder, we provide a simple example. 
+ The forward problem $d = (Id - \alpha\Delta)^{-1}u + \epsilon$; 
+ The inverse problem is that given the data $d$ find the function parameter $u$.  

The **common.py** in the folder of the SimpleSmooth provides the classes 
+ EquSolver: contains forward equation solver, adjoint equation solver, incremental forward solver, and incremental adjoint solver
+ ModelSS: composed by prior measure, differential equations, and noise distributions 

## SteadyStateDarcyFlow
In this folder, we provide codes for inverse problems of the steady state Darcy flow equation. Details of the inverse problems of Darcy flow can be found in a some articles: 
> 1. M. Dashti, A. M. Stuart, The Bayesian Approch to Inverse Problems, Hankbook of Uncertainty Quantification, 2017 [Section 1.3 Elliptic Inverse Problem]
> 2. Junxiong Jia, Peijun Li, Deyu Meng, Stein variational gradient descent on infinite-dimensional space and applications to statistical inverse problems, SIAM Journal on Numerical Analysis, 60(4): 2225-2252, 2022. 

The **common.py** in the folder of the SimpleSmooth provides the classes 
+ EquSolver: contains forward equation solver, adjoint equation solver, incremental forward solver, and incremental adjoint solver
+ ModelDarcyFlow: composed by prior measure, differential equations, and noise distributions 


**Citation:** \
@article{IPBayesML, \
 title = {IPBayesML: Programs for Inverse Problems of PDEs with Bayesian and Machine Learning Methods }, \
 author = {Junxiong Jia}, \
 year = {2022},\
 url = {https://github.com/jjx323/IPBayesML}  \
}
