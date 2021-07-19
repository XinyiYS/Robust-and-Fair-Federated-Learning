import torch
from torch import nn, optim

from utils.models_defined import MNIST_LogisticRegression, MLP_Net, CNN_Net, CNN_Text, ResNet18_torch, CNNCifar_TF


mnist_args = {
	# setting parameters
	'dataset': 'mnist',
	'sample_size_cap': 6000,
	'n_participants': 5,
	'split': 'powerlaw', #or 'classimbalance'

	'batch_size' : 32, 
	'train_val_split_ratio': 0.9,
	'alpha': 0.95,
	'Gamma': 0.5,

	# model parameters
	'model_fn': CNN_Net, #MLP_Net, MNIST_LogisticRegression
	'optimizer_fn': optim.SGD,
	'loss_fn': nn.NLLLoss(), 
	'lr': 0.15,
	'lr_decay':0.977,  #0.977**100 ~= 0.1

	# fairness/training parameters
	'rounds': 60,
	'E': 1,
}



sst_args = {
	# setting parameters
	'dataset': 'sst',
	'n_participants': 5,
	'split': 'powerlaw', #or 'powerlaw' classimbalance
	'batch_size' : 256, 

	'train_val_split_ratio': 0.9,
	'alpha': 0.95,
	'Gamma': 1,
	'lambda': 1, # coefficient between sign_cossim and modu_cossim

	# model parameters
	'model_fn': CNN_Text,
	'embed_num': 20000,
	'embed_dim': 300,
	'class_num': 5,
	'kernel_num': 128,
	'kernel_sizes': [3,3,3],
	'static':False,

	'optimizer_fn': optim.Adam,
	'loss_fn': nn.NLLLoss(), 
	'lr': 1e-4,
	'lr_decay':0.977,  #0.977**100 ~= 0.1

	# training parameters
	'rounds': 100,
	'E': 2,
}


mr_args = {
	# setting parameters
	'dataset': 'mr',
	'n_participants': 5,
	'split': 'powerlaw', #or 'powerlaw' classimbalance

	'batch_size' : 128, 
	'train_val_split_ratio': 0.9,
	'alpha': 0.95,
	'lambda': 0.5, # coefficient between sign_cossim and modu_cossim
	'Gamma':1,

	# model parameters
	'model_fn': CNN_Text,
	'embed_num': 20000,
	'embed_dim': 300,
	'class_num': 2,
	'kernel_num': 128,
	'kernel_sizes': [3,3,3],
	'static':False,

	'optimizer_fn': optim.Adam,
	'loss_fn': nn.NLLLoss(), 
	'lr': 5e-5,
	'lr_decay':0.977,  #0.977**100 ~= 0.1

	# training parameters
	'rounds': 100,
	'E': 2,
}


cifar_cnn_args = {
	# setting parameters
	'dataset': 'cifar10',
	'sample_size_cap': 20000,
	'n_participants': 10,
	'split': 'powerlaw', #or 'classimbalance'

	'batch_size' : 128, 
	'train_val_split_ratio': 0.8,
	'alpha': 0.95,
	'Gamma': 0.15,
	'lambda': 0.5, # coefficient between sign_cossim and modu_cossim

	# model parameters
	'model_fn': CNNCifar_TF, #ResNet18_torch, CNNCifar_TF
	'optimizer_fn': optim.SGD,
	'loss_fn': nn.NLLLoss(),#  nn.CrossEntropyLoss(), 
	'lr': 0.015,
	'lr_decay':0.977,  #0.977**100 ~= 0.1

	# training parameters
	'rounds': 200,
	'E': 1,
}
