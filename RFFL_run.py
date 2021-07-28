import os
from os.path import join as oj
import sys
import json
import copy
from copy import deepcopy as dcopy
import time
import datetime
import random
from collections import defaultdict
from itertools import product
import pickle

import numpy as np
import pandas as pd
import argparse

import torch
from torch import nn, optim
from torch.linalg import norm
from torchtext.data import Batch
import torch.nn.functional as F


from utils.Data_Prepper import Data_Prepper
from utils.arguments import mnist_args, cifar_cnn_args, mr_args, sst_args
from utils.utils import compute_grad_update, add_update_to_model, add_gradient_updates,\
	flatten, unflatten, train_model, evaluate, cosine_similarity, mask_grad_update_by_order

parser = argparse.ArgumentParser(description='Process which dataset to run')
parser.add_argument('-d', '--dataset', help='Dataset name', type=str, required=True)
parser.add_argument('-N', '--participants', help='The number of participants', type=int, default=-1)

# The coefficient to calculate beta = threshold * (1.0/N).
parser.add_argument('-rth', '--threshold', help='Reputation threshold coefficient.', type=float, default=1.0/3.0)


# adversary information
parser.add_argument('-A', '--adversaries', help='The number of adversaries', type=int, default=0)
# five types of attacks: fr - free rider; lf - label-flipping; re - rescaling; vi - value inverting; sr - sign-randomizing; 
parser.add_argument('-atk', '--adversary_attack', help='The type of adversaries', type=str, default='fr', choices=['lf', 're', 'vi', 'sr', 'fr', 'all'])


# by default reputation is enabled, i.e., True
parser.add_argument('-nr', '--no-reputation', dest='reputation', help='Whether to use reputation mechanism: calculating cosine based reputation, and use it for aggregation.', action='store_false')
parser.add_argument('-r','--reputation', dest='reputation', help='Whether to use reputation mechanism: calculating cosine based reputation, and use it for aggregation.', action='store_true')

# by default sparsify is enabled, i.e., True
parser.add_argument('-ns', '--no-sparsify', dest='sparsify', help='Whether to use sparsification: the sparsification weight is reputation (if available), or the local dataset weights.', action='store_false')
parser.add_argument('-s','--sparsify', dest='sparsify', help='Whether to use sparsification: the sparsification weight is reputation (if available), or the local dataset weights.', action='store_true')

parser.add_argument('-cuda', dest='cuda', help='Use cuda if available.', action='store_true')
parser.add_argument('-nocuda', dest='cuda', help='Not to use cuda even if available.', action='store_false')


cmd_args = parser.parse_args()
print(cmd_args)

use_reputation = cmd_args.reputation
use_sparsify = cmd_args.sparsify
N = cmd_args.participants
A = cmd_args.adversaries
atk = cmd_args.adversary_attack
threshold = cmd_args.threshold
R_set = list(range(N+A))


if torch.cuda.is_available() and cmd_args.cuda:
	device = torch.device('cuda')
else:
	device = torch.device('cpu')


if cmd_args.dataset == 'mnist':
	args = copy.deepcopy(mnist_args)

	if N > 0:
		participant_rounds = [[N, N*600]]
	else:
		participant_rounds = [[5, 3000], [10, 6000], [20, 12000]]
	splits = ['uniform', 'classimbalance', 'powerlaw'] 
	args['rounds'] = 100
	args['E'] = 3

elif cmd_args.dataset == 'cifar10':
	args = copy.deepcopy(cifar_cnn_args)	
	participant_rounds = [[10, 20000]]
	splits = ['classimbalance', 'powerlaw', 'uniform']
	args['rounds'] = 200
	args['E'] = 3


# only run with N=5
elif cmd_args.dataset == 'sst':
	args = copy.deepcopy(sst_args)	
	participant_rounds = [[5, 8000]]
	splits = ['powerlaw']
	args['rounds'] = 200
	args['E'] = 3

# only run with N=5	
elif cmd_args.dataset == 'mr':
	args = copy.deepcopy(mr_args)	
	participant_rounds = [[5, 8000]]
	splits = ['powerlaw']
	args['rounds'] = 200
	args['E'] = 3

E = args['E']

for n_participants, sample_size_cap in participant_rounds:
	
	args['n_participants'] = n_participants
	args['sample_size_cap'] = sample_size_cap
	# args['momentum'] = 1.5 / n_participants

	if use_sparsify:
		download_proportions = [0.2, 0.4, 0.6, 1]
	else:
		download_proportions = [1]

	if cmd_args.adversary_attack == 'all':
		atks = ['lf', 're', 'vi', 'sr', 'fr']
	else:
		atks = [cmd_args.adversary_attack]
		
	for atk in atks:

		args['attack'] = atk
		args['n_adversaries'] = A

		for split in splits:
			args['split'] = split #powerlaw ,  classimbalance

			optimizer_fn = args['optimizer_fn']
			loss_fn = args['loss_fn']

			print(args)
			print("Data Split information for honest participants:")
			data_prepper = Data_Prepper(
				args['dataset'], train_batch_size=args['batch_size'], n_participants=args['n_participants'], sample_size_cap=args['sample_size_cap'], 
				train_val_split_ratio=args['train_val_split_ratio'], device=device, args_dict=args)

			valid_loader = data_prepper.get_valid_loader()
			test_loader = data_prepper.get_test_loader()

			train_loaders = data_prepper.get_train_loaders(args['n_participants'], args['split'])
			shard_sizes = data_prepper.shard_sizes

			adv_loaders = []
			if A > 0:
				print("Data Split information for adversaries:")

				# adversarial loaders follows the uniform split, and each has 600 examples
				adv_data_prepper = Data_Prepper(
					args['dataset'], train_batch_size=args['batch_size'], n_participants=A, sample_size_cap= 600 * A, 
					train_val_split_ratio=args['train_val_split_ratio'], device=device, args_dict=args)
				adv_loaders = adv_data_prepper.get_train_loaders(A, 'uniform')	
				shard_sizes +=  adv_data_prepper.shard_sizes

			shard_sizes = torch.tensor(shard_sizes).float()
			relative_shard_sizes = torch.div(shard_sizes, torch.sum(shard_sizes))           
			print("Number of honest participants: {}. Number of adversaries: {}, type: {}.".format(N, A, atk))
			print("Shard sizes are: ", shard_sizes.tolist())

			if args['dataset'] in ['mr', 'sst']:
				server_model = args['model_fn'](args=data_prepper.args).to(device)
			else:
				server_model = args['model_fn']().to(device)

			D = sum([p.numel() for p in server_model.parameters()])

			# ---- init honest participants ----
			participant_models, participant_optimizers, participant_schedulers = [], [], []

			for i in range(N):
				model = copy.deepcopy(server_model)
				optimizer = optimizer_fn(model.parameters(), lr=args['lr'])
				scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = args['lr_decay'])

				participant_models.append(model)
				participant_optimizers.append(optimizer)
				participant_schedulers.append(scheduler)

			# ---- init adversaries ----
			adv_models, adv_optimizers, adv_schedulers = [], [], []
			for i in range(A):
				model = copy.deepcopy(server_model)
				optimizer = optimizer_fn(model.parameters(), lr=args['lr'])
				scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = args['lr_decay'])

				adv_models.append(model)
				adv_optimizers.append(optimizer)
				adv_schedulers.append(scheduler)

			rs = torch.zeros(N+A, device=device)
			past_phis = []

			adv_lf_perfs = defaultdict(list) # only used for label-flipping adversaries

			valid_perfs = defaultdict(list) # validation performance results
			local_perfs = defaultdict(list) # local training dataset performance results

			rs_dict = []
			r_threshold = []

			qs_dict = []

			# ---- FL begins ---- 
			for _round in range(args['rounds']):

				gradients = []

				# ---- Adversaries if A != 0---- 
				for i in range(A):
					loader = adv_loaders[i]
					model = adv_models[i]
					optimizer = adv_optimizers[i]
					scheduler = adv_schedulers[i]

					if atk == 'fr':
						adv_grad = []
						for param in model.parameters():
							adv_grad.append(  (torch.rand(param.size()) * 2 -1).to(device)  )
								# torch.from_numpy(np.random.choice([-1, 1], size=param.shape)).to(device) )
					
					elif atk == 'lf':
						model_before = dcopy(model)
						for epoch in range(E):
							for i, batch in enumerate(loader):
								if isinstance(batch, Batch):
									batch_data, batch_target = batch.text, batch.label
									batch_data = batch_data.permute(1, 0)
								else:
									batch_data, batch_target = batch[0], batch[1]
								
								batch_data, batch_target = batch_data.to(device), batch_target.to(device)

								from_to = ['1', '7']
								# self.adversary.split('_')[1].split('-')
								from_target, to_target = int(from_to[0]), int(from_to[1])

								for i, target in enumerate(batch_target):
									batch_target[i] = to_target if target == from_target else target
							
								optimizer.zero_grad()
								loss_fn(model(batch_data), batch_target).backward()									
								optimizer.step()
								scheduler.step()

						adv_grad = compute_grad_update(model_before, model)

					elif atk in ['sr', 'vi', 're']:

						model_before = dcopy(model)
						model = train_model(model, loader, loss_fn, optimizer, device=device, E=E, scheduler=scheduler)

						adv_grad = compute_grad_update(model_before, model)

						if atk == 're':
							noise = 10
							# multiply by a random noise from [-noise, noise] of the elements in gradient, element-wise
							for grad in adv_grad:
								grad.data *= (noise * 2) * torch.rand(size=grad.shape, device=grad.device) - noise
								grad.data *= noise

						elif atk == 'sr':
							# randomly flip the signs of the elements in gradient, element-wise
							for grad in adv_grad:
								grad.data *= torch.from_numpy(np.random.choice([-1, 1], size=grad.shape)).to(grad.device)
								# (torch.rand(size=grad.shape, device=grad.device) - 0.5).sign()

						elif atk == 'vi':
							# randomly flip inverts the elements in gradient, element-wise
							for grad in adv_grad:
								random_bits = torch.rand(size=grad.shape, device=grad.device) < 0.5
								grad.data[random_bits] = (grad.data[random_bits] + 1e-10).pow_(-1)

					flattened = flatten(adv_grad)
					norm_value = norm(flattened) + 1e-7 # to prevent division by zero
					if norm_value > args['Gamma']:
						adv_grad = unflatten(torch.multiply(torch.tensor(args['Gamma']), torch.div(flattened,  norm_value)), adv_grad)

					gradients.append(adv_grad)



				# ---- Honest participants ---- 
				for i in range(N):
					loader = train_loaders[i]
					model = participant_models[i]
					optimizer = participant_optimizers[i]
					scheduler = participant_schedulers[i]

					model.train()
					model = model.to(device)

					backup = copy.deepcopy(model)

					model = train_model(model, loader, loss_fn, optimizer, device=device, E=E, scheduler=scheduler)

					gradient = compute_grad_update(old_model=backup, new_model=model, device=device)
					
					flattened = flatten(gradient)
					norm_value = norm(flattened) + 1e-7 # to prevent division by zero
					if norm_value > args['Gamma']:
						gradient = unflatten(torch.multiply(torch.tensor(args['Gamma']), torch.div(flattened,  norm_value)), gradient)

						model.load_state_dict(backup.state_dict())
						add_update_to_model(model, gradient, device=device)

					gradients.append(gradient)


				# ---- Server Aggregate ----

				aggregated_gradient =  [torch.zeros(param.shape).to(device) for param in server_model.parameters()]

				if not use_reputation:
					# fedavg
					for gradient, weight in zip(gradients, relative_shard_sizes):
						add_gradient_updates(aggregated_gradient, gradient, weight=weight)

				else:
					if _round == 0:
						weights = torch.div(shard_sizes, torch.sum(shard_sizes))
					else:
						weights = rs

					for gradient, weight in zip(gradients, weights):
						add_gradient_updates(aggregated_gradient, gradient, weight=weight)

					flat_aggre_grad = flatten(aggregated_gradient)

					phis = torch.zeros( N + A, device=device)
					for i, gradient in enumerate(gradients):
						phis[i] = F.cosine_similarity(flatten(gradient), flat_aggre_grad, 0, 1e-10) 

					past_phis.append(phis)

					rs = args['alpha'] * rs + (1-args['alpha']) * phis
					for i in range(N + A):
						if i not in R_set:
							rs[i] = 0
					rs = torch.div(rs, rs.sum())

					# --- reputation threshold
					# start removing participants only after 10 rounds
					if _round >= 10:
						R_set_copy = dcopy(R_set)
						curr_threshold = threshold * (1.0/ len(R_set_copy))

						for i in range(N + A):
							# only operation is to remove a reputable participant, if necessary. All others left untouched.
							if i in R_set_copy and rs[i] < curr_threshold:
								rs[i] = 0
								R_set.remove(i)
								print("---- in round {} removing {}. ".format(_round, i))


					rs = torch.div(rs, rs.sum())
					r_threshold.append( threshold * (1.0 / len(R_set)) )
					q_ratios = torch.div(rs, torch.max(rs))
					
					rs_dict.append(rs)
					qs_dict.append(q_ratios)
				

				for i in range(N+A):

					if use_sparsify and use_reputation:

						q_ratio = q_ratios[i]
						reward_gradient = mask_grad_update_by_order(aggregated_gradient, mask_percentile=q_ratio, mode='layer')

					elif use_sparsify and not use_reputation:
						
						# relative_shard_sizes[i] the relative dataset weight of the local dataset
						reward_gradient = mask_grad_update_by_order(aggregated_gradient, mask_percentile=relative_shard_sizes[i], mode='layer')

					else: # not use_sparsify
						# the reward_gradient is the whole gradient
						reward_gradient = aggregated_gradient

					if i < N:
						add_update_to_model(participant_models[i], reward_gradient)
					else: # N<= i < N+A:
						add_update_to_model(adv_models[i-N], reward_gradient)

				for i, model in enumerate(participant_models):
					loss, accuracy = evaluate(model, valid_loader, loss_fn=loss_fn, device=device)
					if A > 0 and  atk == 'lf':
						loss, target_accuracy, attack_success_rate = evaluate(model, valid_loader, loss_fn=loss_fn, device=device, label_flip='1-7')

						adv_lf_perfs[str(i)+'_target_accu'].append(target_accuracy.item())
						adv_lf_perfs[str(i)+'_attack_success'].append(attack_success_rate.item())
						

					valid_perfs[str(i)+'_loss'].append(loss.item())
					valid_perfs[str(i)+'_accu'].append(accuracy.item())

					fed_loss, fed_accu = 0, 0
					for j, train_loader in enumerate(train_loaders):
						loss, accuracy = evaluate(model, train_loader, loss_fn=loss_fn, device=device)

						if j == i:
							local_perfs[str(i)+'_loss'].append(loss.item())
							local_perfs[str(i)+'_accu'].append(accuracy.item())


			# ---- Results saving ---- 
			participant_str = '{}-{}-{}'.format(
				args['split'][:3].upper(),
				'N'+str(N),
				'A'+str(A)+atk)

			folder = oj('RFFL_results', 
				args['dataset'], 
				participant_str, 
				'r{}s{}'.format(str(int(use_reputation)), str(int(use_sparsify)) ) )

			os.makedirs(folder, exist_ok=True)

			if use_reputation:
				rs_dict = torch.stack(rs_dict).detach().cpu().numpy()
				df = pd.DataFrame(rs_dict)
				df.to_csv(oj(folder, 'rs.csv'), index=False)


				qs_dict = torch.stack(qs_dict).detach().cpu().numpy()
				df = pd.DataFrame(qs_dict)
				df.to_csv(oj(folder, 'qs.csv'), index=False)


			# print("Past phis:", (torch.stack(past_phis)).reshape(-1, args['n_participants']))

			df = pd.DataFrame(valid_perfs)
			# print("Validation performance:")
			df.to_csv(oj(folder, 'valid.csv'), index=False)
			
			if A > 0 and  atk == 'lf':
				df = pd.DataFrame(adv_lf_perfs)
				df.to_csv(oj(folder, 'adv_lf.csv'), index=False)

			df = pd.DataFrame(local_perfs)
			df.to_csv(oj(folder, 'local.csv'), index=False)


			with open(oj(folder, 'settings_dict.txt'), 'w') as file:
				[file.write(key + ' : ' + str(value) + '\n') for key, value in args.items()]

			with open(oj(folder, 'settings_dict.pickle'), 'wb') as f: 
				pickle.dump(args, f)
			 
			# pickle - loading 
			# with open(oj(folder,'settings_dict.pickle'), 'rb') as f: 
				# args = pickle.load(f) 

