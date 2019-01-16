# coding=utf-8

import os

class Config:
	'''
	some constants for training and modeling
	'''
	def __init__(self):
		self.data_root =os.path.join(os.getcwd(),"LCSP_data")
		if not os.path.exists(self.data_root):
			os.makedirs(self.data_root)

	'''''''''' path to save data '''''''''''

	@property
	def DATA_LIST_PATH(self):
		'''
		path for training data
		'''
		return os.path.join(config.data_root,"fake_data_list.pkl")  # this is fake data.

	@property
	def TEST_DATA_PATH(self):
		'''
		path for training data
		'''
		return os.path.join(config.data_root,"fake_test_data.pkl") # this is fake data.

	@property
	def CHECKPOINT_DIR(self):
		'''
		Directory to saving model checkpoints
		'''
		dir=os.path.join(self.data_root,"checkpoints")
		if not os.path.exists(dir):
			os.makedirs(dir)
		return dir

	@property
	def LOG_DIR(self):
		'''
		Directory to saving logs, e.g. save loss curve when training
		'''
		dir=os.path.join(self.data_root,"logs")
		if not os.path.exists(dir):
			os.makedirs(dir)
		return dir

	@property
	def BEST_MODEL_PATH(self):
		'''
		path to save model that have best evaluation results
		'''
		return os.path.join(config.CHECKPOINT_DIR, "best_model.pkl")

	@property
	def TRAIN_LOG(self):
		'''
		path to save logs for training step
		'''
		return os.path.join(config.LOG_DIR, "train_loss.log")

	@property
	def EVAL_LOG(self):
		'''
		path to save logs for evaluation step
		'''
		return os.path.join(config.LOG_DIR, "eval_loss.log")

	@property
	def key2ids_path(self):
		'''
		path fot key2ids_dict
		'''
		return os.path.join(config.data_root, "key2ids.dict")


	'''''''''' model parameters '''''''''''

	@property
	def FEATURE_DIM(self):
		'''
		dimension of raw - features (exluded time step)
		'''
		return 20

	@property
	def HIDDEN_SIZE(self):
		'''
		dimension of hidden size for lstm module
		'''
		return 128

	@property
	def NUM_IDS(self):
		'''
		number of unique RV-key (key-index)
		'''
		return 1000

	@property
	def EMBED_DIM(self):
		'''
		dimension of output embedding vector
		'''
		return 128

	@property
	def USE_BIDIRECTIONAL(self):
		'''
		if True, use bidirectional-lstm
		'''
		return True

	'''''''''' training parameters '''''''''''

	@property
	def EPOCHS(self):
		'''
		number of epochs for training
		'''
		return 10

	@property
	def BATCH_SIZE(self):
		'''
		using minibatch size
		'''
		return 2

	@property
	def EVAL_PER_BATCH(self):
		'''
		number of evaluation for each batch
		'''
		return 2

	@property
	def LEARNING_RATE(self):
		'''
		learning rate for 'Adam'  or 'SGD'
		'''
		return 0.001


config = Config()