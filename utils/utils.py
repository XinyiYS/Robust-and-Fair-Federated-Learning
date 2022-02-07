import math
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchtext.data import Batch

import torch.nn.functional as F


def compute_grad_update(old_model, new_model, device=None):
	# maybe later to implement on selected layers/parameters
	if device:
		old_model, new_model = old_model.to(device), new_model.to(device)
	return [(new_param.data - old_param.data) for old_param, new_param in zip(old_model.parameters(), new_model.parameters())]

def add_gradient_updates(grad_update_1, grad_update_2, weight = 1.0):
	assert len(grad_update_1) == len(
		grad_update_2), "Lengths of the two grad_updates not equal"
	
	for param_1, param_2 in zip(grad_update_1, grad_update_2):
		param_1.data += param_2.data * weight


def add_update_to_model(model, update, weight=1.0, device=None):
	if not update: return model
	if device:
		model = model.to(device)
		update = [param.to(device) for param in update]
			
	for param_model, param_update in zip(model.parameters(), update):
		param_model.data += weight * param_update.data
	return model

def compare_models(model1, model2):
	for p1, p2 in zip(model1.parameters(), model2.parameters()):
		if p1.data.ne(p2.data).sum() > 0:
			return False # two models have different weights
	return True


def sign(grad):
	return [torch.sign(update) for update in grad]

def flatten(grad_update):
	return torch.cat([update.data.view(-1) for update in grad_update])

def unflatten(flattened, normal_shape):
	grad_update = []
	for param in normal_shape:
		n_params = len(param.view(-1))
		grad_update.append(torch.as_tensor(flattened[:n_params]).reshape(param.size())  )
		flattened = flattened[n_params:]

	return grad_update

def l2norm(grad):
	return torch.sqrt(torch.sum(torch.pow(flatten(grad), 2)))

def cosine_similarity(grad1, grad2, normalized=False):
	"""
	Input: two sets of gradients of the same shape
	Output range: [-1, 1]
	"""

	cos_sim = F.cosine_similarity(flatten(grad1), flatten(grad2), 0, 1e-10) 
	if normalized:
		return (cos_sim + 1) / 2.0
	else:
		return cos_sim


from math import pi
def angular_similarity(grad1, grad2):
	return 1 - torch.div(torch.acovs(cosine_similarity(grad1, grad2)), pi)

def evaluate(model, eval_loader, device, loss_fn=None, verbose=False, label_flip=None):
	model.eval()
	model = model.to(device)
	correct = 0
	total = 0
	loss = 0

	target_correct = 0
	target_total = 0
	attack_success = 0

	with torch.no_grad():
		for i, batch in enumerate(eval_loader):

			if isinstance(batch, Batch):
				batch_data, batch_target = batch.text, batch.label
				# batch_data.data.t_(), batch_target.data.sub_(1)  # batch first, index align
				batch_data = batch_data.permute(1, 0)
			else:
				batch_data, batch_target = batch[0], batch[1]

			batch_data, batch_target = batch_data.to(device), batch_target.to(device)
			outputs = model(batch_data)

			if loss_fn:
				loss += loss_fn(outputs, batch_target)
			else:
				loss = None
			correct += (torch.max(outputs, 1)[1].view(batch_target.size()).data == batch_target.data).sum()
			total += len(batch_target)

			if label_flip:
				classes = label_flip.split('-')
				from_class, to_class = int(classes[0]), int(classes[1])
				indices = batch_target==from_class
				target_total += sum(indices)

				target_class_outputs, target_class_labels = outputs[indices], batch_target[indices]
				target_correct += (torch.max(target_class_outputs, 1)[1].view(target_class_labels.size()).data == target_class_labels.data).sum()

				attack_success += (torch.max(target_class_outputs, 1)[1].view(target_class_labels.size()).data == to_class).sum()

		accuracy =  correct.float() / total
		if loss_fn:
			loss /= total

		if label_flip:
			target_accuracy = target_correct.float() / target_total
			attack_success_rate =  attack_success.float()/target_total.item()
			# print("For attack class: ", label_flip)
			# print("Correct:", target_correct.item(), "Total: ", target_total.item(), "Accuracy: ", target_accuracy.item())
			# print("Successful attacks:", attack_success.item(), "Attack success rate:",)
			return loss, target_accuracy, attack_success_rate
	
	if verbose:
		print("Loss: {:.6f}. Accuracy: {:.4%}.".format(loss, accuracy))
	return loss, accuracy

from torchtext.data import Batch
def train_model(model, loader, loss_fn, optimizer, device, E=1, **kwargs):

	model.train()
	for e in range(E):
		# running local epochs
		for _, batch in enumerate(loader):
			if isinstance(batch, Batch):
				data, label = batch.text, batch.label
				data = data.permute(1, 0)
				# data.data.t_(), label.data.sub_(1)  # batch first, index align
			else:
				data, label = batch[0], batch[1]

			data, label = data.to(device), label.to(device)

			optimizer.zero_grad()
			pred = model(data)
			loss_fn(pred, label).backward()

			if 'masks' in kwargs:
				# set the grad for the parameters in the mask to be zero
				for zero_mask, layer in zip(kwargs['masks'], model.parameters()):				
					if len(zero_mask) == 0:
						print("zero mask is empty")
						print("original grad is:", layer.grad)

					copy = layer.grad.view(-1)
					copy.data[zero_mask] = 0
					layer.grad.data = copy.reshape(layer.grad.size())
					if len(zero_mask) == 0:
						print("zero mask is empty")
						print("masked grad is:", layer.grad)

			optimizer.step()

		if 'scheduler' in kwargs: kwargs['scheduler'].step()
	
	return model



def mask_grad_update_by_order(grad_update, mask_order=None, mask_percentile=None, mode='all'):

	if mode == 'all':
		# mask all but the largest <mask_order> updates (by magnitude) to zero
		all_update_mod = torch.cat([update.data.view(-1).abs()
									for update in grad_update])
		if not mask_order and mask_percentile is not None:
			mask_order = int(len(all_update_mod) * mask_percentile)
		
		if mask_order == 0:
			return mask_grad_update_by_magnitude(grad_update, float('inf'))
		else:
			topk, indices = torch.topk(all_update_mod, mask_order)
			return mask_grad_update_by_magnitude(grad_update, topk[-1])

	elif mode == 'layer': # layer wise largest-values criterion
		grad_update = copy.deepcopy(grad_update)

		mask_percentile = max(0, mask_percentile)
		for i, layer in enumerate(grad_update):
			layer_mod = layer.data.view(-1).abs()
			if mask_percentile is not None:
				mask_order = math.ceil(len(layer_mod) * mask_percentile)

			if mask_order == 0:
				grad_update[i].data = torch.zeros(layer.data.shape, device=layer.device)
			else:
				topk, indices = torch.topk(layer_mod, min(mask_order, len(layer_mod)-1))																																												
				grad_update[i].data[layer.data.abs() < topk[-1]] = 0
		return grad_update

def mask_grad_update_by_magnitude(grad_update, mask_constant):

	# mask all but the updates with larger magnitude than <mask_constant> to zero
	# print('Masking all gradient updates with magnitude smaller than ', mask_constant)
	grad_update = copy.deepcopy(grad_update)
	for i, update in enumerate(grad_update):
		grad_update[i].data[update.data.abs() < mask_constant] = 0
	return grad_update


import numpy as np
np.random.seed(1111)


def random_split(sample_indices, m_bins, equal=True):
	sample_indices = np.asarray(sample_indices)
	if equal:
		indices_list = np.array_split(sample_indices, m_bins)
	else:
		split_points = np.random.choice(
			n_samples - 2, m_bins - 1, replace=False) + 1
		split_points.sort()
		indices_list = np.split(sample_indices, split_points)

	return indices_list