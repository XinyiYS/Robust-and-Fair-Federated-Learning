# Robust-and-Fair-Federated-Learning
Implementing the algorithm from our paper: "A Reputation Mechanism Is All You Need: Collaborative Fairness and Adversarial Robustness in Federated Learning".


>📋 In this work, we propose a Robust Fair Federated Learning (RFFL) framework to simultaneously achieve adversarial robustness and collaborative fairness in Federated learning by using a reputation mechanism.


## Citing
If you have found our work to be useful in your work, please consider citing it with the following bibtex:
```
@InProceedings{Xu2021RFFL,
    title={A Reputation Mechanism Is All You Need: Collaborative Fairness and Adversarial Robustness in Federated Learning},
    author={Xinyi Xu and Lingjuan Lyu},
    year={2021}
    booktitle={International Workshop on Federated Learning for User Privacy and Data Confidentiality in Conjunction with ICML 2021 (FL-ICML'21)},
}
```



## Requirements

To install requirements:

```setup
conda env create -f environment.yml
```
>📋  We recommend managing your environment using Anaconda, both for the versions of the packages used here and for easy management. 

>📋  Our code automatically detects GPU(s) through NVIDIA driver, and if not available it will use CPU instead.



## Running the script

To run the code in the paper, run this command:
```
python RFFL_run.py -d mnist -N 10 -A 2 -cuda
```
The above command means to run MNIST dataset, with 10 honest participants and 2 adversaries.

>📋  Note there are several command-line arguments which can be found in `RFFL_run.py`.

>📋  Running `RFFL_run.py` starts the experiments specified by the arguments and it creates and writes to corresponding directories.
