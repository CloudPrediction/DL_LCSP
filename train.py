# coding=utf-8

import os
import sys
import logging
from sklearn.externals import joblib
from sklearn import metrics
from sklearn.utils import shuffle
import numpy as np
import torch
import time

from models import LCSP_LSTM, LCSP_LSTM_Attn, LCSP_LSTM_Emb, LCSP_Net
from config import config as c

import warnings
warnings.filterwarnings('ignore')

# init device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

'''''''''''''''Train and evaluate the model  '''''''''''''''


# evaluate function
def evaluate(model, x, key_ids, y, loss_fn):
	"""
	Evaluation function for training. Evaluation metrics include f1, precesion, recall, AUC .
	:param model: pytorch-model, model for training
	:param x: numpy array, temporal features input of model for evaluation
	:param key_ids: array-like, RV-key index of each sample
	:param y: numpy array, target of model
	:param loss_fn: pytorch-loss-function
	:return res: dict, evaluation results containing f1,precision, recall, threshold, AUC, loss
	"""
	x=torch.tensor(x, dtype=torch.float).to(device)
	key_ids=torch.tensor(key_ids, dtype=torch.long).to(device)
	if model.__class__.__name__ == 'LCSP_LSTM_Emb':
		output, v_k = model(x, key_ids)
	elif model.__class__.__name__ == 'LCSP_LSTM_Attn':
		output, w = model(x)
	elif model.__class__.__name__ == 'LCSP_LSTM':
		output = model(x)
	else:
		output, v_k, w = model(x, key_ids)
	output=output.cpu()
	loss = loss_fn(output, torch.tensor(y, dtype=torch.float)).detach().numpy()
	output = output.flatten().detach().numpy()
	res = {'loss': loss, 'f1': 0, 'precision': 0, 'recall': 0, 'threshold': 0, 'auc': 0}
	for thr in np.linspace(0, 1, 1001):
		t_pred = [int(x >= thr) for x in output]
		f1 = metrics.f1_score(y, t_pred)
		if f1 > res['f1']:  # if current f1 is greater than previous one, update the metrics
			res['precesion'] = metrics.precision_score(y, t_pred)
			res['recall'] = metrics.recall_score(y, t_pred)
			res['f1'] = f1
	res['auc'] = metrics.roc_auc_score(y, output)
	return res


# train function
def train(model, train_x, train_ids, train_y, valid_x, valid_ids, valid_y, save_best=True, write_log=True):
	"""
	train the model.
	:param model: pytorch-model, model for training
	:param train_x: numpy array, temporal features input of training
	:param train_ids: array-like, RV-key index of each training sample
	:param train_y: numpy array, target of training  data
	:param valid_x: numpy array, temporal features input of validation
	:param valid_ids: array-like, RV-key index of each validation sample
	:param valid_y: numpy array, target of validation  data
	:param save_best: bool, if True , only save the one model that have best performance in validation data;
	if False, save model every validation step.
	:param write_log: bool, if True , write loss and metrics in the .log file every evaluation step.
	:return model: pytorch-model, return trained model
	:return best_res: dict, best evaluation results for model,
	contaning f1,precision, recall, threshold, AUC, loss, epoch, iteration
	"""
	n_iters = int(len(train_x) / c.BATCH_SIZE)
	eval_every = int(n_iters / c.EVAL_PER_BATCH)

	print('number of iteration per batch:{0}, evaluation every {1} iterations.'.format(n_iters, eval_every))

	# init
	loss_fn = torch.nn.BCELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=c.LEARNING_RATE)#0.001
	# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.7)
	# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

	# train
	best_res = {'f1': 0, 'precision': 0, 'recall': 0, 'threshold': 0}
	st = time.time()
	for epoch in range(1, c.EPOCHS + 1):
		train_x, train_y = shuffle(train_x, train_y, random_state=7)
		for i in range(n_iters):
			x = torch.tensor(train_x[i * c.BATCH_SIZE: (i + 1) * c.BATCH_SIZE], dtype=torch.float).to(device)
			key_ids = torch.tensor(train_ids[i * c.BATCH_SIZE: (i + 1) * c.BATCH_SIZE], dtype=torch.long).to(
				device)
			target = torch.tensor(train_y[i * c.BATCH_SIZE: (i + 1) * c.BATCH_SIZE], dtype=torch.float).to(device)
			if model.__class__.__name__ =='LCSP_LSTM_Emb':
				output, v_k = model(x, key_ids)
			elif model.__class__.__name__ =='LCSP_LSTM_Attn':
				output, w = model(x)
			elif model.__class__.__name__ == 'LCSP_LSTM':
				output = model(x)
			else:
				output,v_k, w = model(x, key_ids)
			loss = loss_fn(output, target)
			if write_log:
				with open(c.TRAIN_LOG, "a") as f:
					f.write("{0},{1},{2}\n".format(epoch, i, loss))
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if i % eval_every == 0:
				print("\n epoch: {0}, iter: {1} train_loss: {2}, time_cost: {3}".format(
					epoch, i, loss.cpu().detach().numpy(), time.time() - st))
				st = time.time()
				res = evaluate(model, valid_x, valid_ids, valid_y, loss_fn)
				if write_log:
					with open(c.EVAL_LOG, "a") as f:
						f.write("{0},{1},{2},{3},{4},{5},{6},{7}\n".format(epoch, i, res['loss'], res['f1'],
						                                                   res['precision'], res['recall'],
						                                                   res['threshold'], res['auc']))
				# scheduler.step(loss)
					print("test loss: {0} test_f1: {1}, test_auc: {2}, time_cost: {3}".format(
					res['loss'], res['f1'], res['auc'], time.time() - st))
				if res['f1'] > best_res['f1']:
					best_res = res
					best_res['epoch'] = epoch
					best_res['iter'] = iter
					if save_best:
						torch.save(model, c.BEST_MODEL_PATH)
				if not save_best:
					torch.save(model, os.path.join(c.CHECKPOINT_DIR,
					                               '{0}_{1}_{2}_{3}.pkl'.format(model.__class__.__name__, epoch, i,
					                                                            res['loss'])))

				st = time.time()

	print('Best performance for model:{0}, test_f1:{1}, test_auc: {2}, epoch: {3}, iter: {4}'.format(
		model.__class__.__name__, best_res['f1'], best_res['auc'], best_res['epoch'], best_res['iter']))
	return model, best_res


''''''''''''''' Get data and train model on several data sets  '''''''''''''''


def get_data(data_list_path):
	"""
	get the training data and build key2ids_dict from data .
	:param data_path: path of data file
	:return data_list: list or array like, list of data,
	each element of the data_list is a dataset for some specific dates,
	e.g. data_set[0] containing the training set of 2018-01-01~2018-03-31, validation data of 2018-04-01~2018-04-15
	:return key2ids_dict:
	"""
	# this function can be changed according to the user's needs of reading data
	data_list = joblib.load(data_list_path)
	RV_key = set()
	for dat in data_list:
		train_x = dat['train_x']
		train_key = dat['train_key']
		train_y = dat['train_y']
		valid_x = dat['valid_x']
		valid_key = dat['valid_key']
		valid_y = dat['valid_y']
		date = dat['Date']
		print(str(date))
		print('train:', train_x.shape, train_y.shape, len(train_key), 'test:', valid_x.shape, valid_y.shape,
		             len(valid_key))
		RV_key |= set(train_key)
		RV_key |= set(valid_key)
	key2ids_dict = {'unk': 0}
	for ind, key in enumerate(sorted(RV_key), 1):
		key2ids_dict[key] = ind
	print('number of all unique RV-keys:{0}'.format(len(key2ids_dict)))

	#joblib.dump(key2ids_dict,c.key2ids_path)
	#key2ids_dict=joblib.load(c.key2ids_path)

	return data_list, key2ids_dict


def train_model_on_several_data(model, data_list, key2ids_dict):
	"""
	train one model in several dataset. each dataset is splitted by date time.
	:param model: pytorch-model, model for training
	:param data_list: list or array like, list of data,
	each element of the data_list is a dataset for some specific dates,
	e.g. data_set[0] containing the training set of 2018-01-01~2018-03-31, validation data of 2018-04-01~2018-04-15
	:param key2ids_dict: dict, convert RV-key(string) to index (int). e.g. {'Dsv3':1,...}
	:return score_dict: dict, dictionary of each dataset evaluation results
	"""
	# this function can be changed according to the user's needs
	score_dict = {}  # {dataset: [scores]}
	for dat in data_list:
		print('\n====================== Dataset for Date: ', dat['Date'], '\n')
		train_x = np.array(dat['train_x'])
		train_y = np.array(dat['train_y'])
		valid_x = np.array(dat['valid_x'])
		valid_y = np.array(dat['valid_y'])
		print('train:', train_x.shape, train_y.shape, 'test:', valid_x.shape, valid_y.shape)
		print('positive negative ratio in train data:{0}, in validation data:{1}', sum(train_y) / len(train_y),
		             sum(valid_y) / len(valid_y))
		train_ids = [key2ids_dict.get(r, 0) for r in dat['train_key']]
		valid_ids = [key2ids_dict.get(r, 0) for r in dat['valid_key']]
		print('--')

		trained_model, score = train(model, train_x, train_ids, train_y, valid_x, valid_ids, valid_y)
		score_dict[dat['Date']] = score
		print(
			'Evaluation of model {0}, for date {1} , f1: {2}, auc:{3} '.format(model.__class__.__name__, dat['Date'],
			                                                                   score['f1'], score['auc']))
	return score_dict


''''''''''''''' main  '''''''''''''''


def main():
	data_list, key2ids_dict = get_data(data_list_path=c.DATA_LIST_PATH)
	models_list = ['LCSP_LSTM','LCSP_LSTM_Attn','LCSP_LSTM_Emb','LCSP_Net']
	model_name = 'LCSP_Net' #  Can be choose one of models
	model = eval(model_name+'()').to(device)
	score_dict = train_model_on_several_data(model=model, data_list=data_list,
	                                         key2ids_dict=key2ids_dict)


if __name__ == '__main__':
	main()
