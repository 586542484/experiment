#### Folder content description：

1. `code folder`：

   The folder contains our experimental code, and in the experiment, we utilized the English dataset AL-CPL and the Chinese dataset ZH-AL-CPL. Concerning the parameter settings in the experiment, there are two types of parameters: threshold parameters and hyperparameters. Threshold parameters include the semantic relevance threshold and the threshold for the reconstruction matrix output by VGAE, and we set both of them to 0.6. Hyperparameters consist of the learning rate, weight decay, and the number of heads in the multi-head attention mechanism. After conducting multiple experiments for parameter tuning, we found that the best experimental results were achieved with a learning rate of 0.0001, weight decay of 0.04, and three heads in the multi-head attention mechanism. Therefore, we adopted these values as the hyperparameter settings for the experiment.

2. `dataset folder`：

   This folder contains the data set we used. Detailed information about the data set can be viewed in the data set section of the full paper.

3. `full paper folder`：

   The files in this folder comprise the full version of our research paper.
