# coding=utf-8

import torch
import torch.nn.functional as F
from config import config as c


''''''''''''''' Mechanisms  '''''''''''''''


class Classifier(torch.nn.Module):
	"""
	Final classifier for low capacity status prediction problem.
	This classifier was build with fully connected layer.
    :param input_size: int, input dimension of first fully connected layer
	"""

	def __init__(self, input_size=c.HIDDEN_SIZE):
		super(Classifier, self).__init__()
		self.fc1 = torch.nn.Linear(input_size, 128)
		self.fc2 = torch.nn.Linear(128, 32)
		self.fc3 = torch.nn.Linear(32, 1)

	def forward(self, x):
		x = F.relu(F.dropout(self.fc1(x), 0.5))
		x = F.relu(F.dropout(self.fc2(x), 0.5))
		x = torch.sigmoid(self.fc3(x))
		return x


class Embedding_Layer(torch.nn.Module):
	"""
	Embedding layer (include dimension align) for low capacity status prediction problem.
	This layer mapped RV-key index to dense vector. e.g. [1] -> [0.6,0.3,0.1,....,0.8]
	:param num_embeddings: int, number of unique RV-key (key-index)
	:param embed_dim: int, dimension of output embedding vector
	"""

	def __init__(self, num_embeddings=c.NUM_IDS, embed_dim=c.EMBED_DIM,align_dim=c.HIDDEN_SIZE):
		super(Embedding_Layer, self).__init__()
		self.emb = torch.nn.Embedding(num_embeddings=num_embeddings,
		                              embedding_dim=embed_dim)
		self.emb2 = torch.nn.Linear(embed_dim, align_dim)

	def forward(self, key):
		v_k = self.emb(key)
		v_k = self.emb2(v_k)
		return v_k


class Attention_Layer(torch.nn.Module):
	"""
	Attention layer for low capacity status prediction problem.
	This layer  output the importance of features on different days.
	Input of this layer is the LSTM' all hidden states.
	:param input_size: int, input dimension of first fully connected layer
	"""

	def __init__(self, input_size=c.HIDDEN_SIZE):
		super(Attention_Layer, self).__init__()
		self.attn_u1 = torch.nn.Linear(input_size, 64)
		self.attn_u2 = torch.nn.Linear(64, 1)

	def forward(self, x):
		w = self.attn_u1(x)
		w = self.attn_u2(w)
		w = F.softmax(w, dim=1)
		return w


''''''''''''''' Base LSTM model '''''''''''''''


class LCSP_LSTM(torch.nn.Module):
	"""
	Base LSTM model for low capacity status prediction problem.
	This base model include lstm and classifier module.
	:param input_size: int, dimension of raw - features (exluded time step)
	:param hidden_size: int, dimension of hidden size for lstm module
	:param bidirectional: bool, if True, use bidirectional-lstm
	"""

	def __init__(self, input_size=c.FEATURE_DIM, hidden_size=c.HIDDEN_SIZE, bidirectional=c.USE_BIDIRECTIONAL):
		super(LCSP_LSTM, self).__init__()
		self.lstm = torch.nn.LSTM(input_size=input_size,
		                          hidden_size=hidden_size,
		                          batch_first=True,
		                          bidirectional=bidirectional)
		if bidirectional:
			hidden_size = hidden_size * 2
		self.clf = Classifier(input_size=hidden_size)

	def forward(self, x):
		o, (h_T, c_T) = self.lstm(x)
		h = torch.cat([h_T[0], h_T[1]], dim=1)
		proba = self.clf(h)
		return proba


''''''''''''''' LSTM + Attention '''''''''''''''


class LCSP_LSTM_Attn(torch.nn.Module):
	"""
	LSTM with attention mechanism for low capacity status prediction problem.
	This model include lstm, attention layer and classifier module.
	:param input_size: int, dimension of raw - features (exluded time step)
	:param hidden_size: int, dimension of hidden size for lstm module
	:param bidirectional: bool, if True, use bidirectional-lstm
	"""

	def __init__(self, input_size=c.FEATURE_DIM, hidden_size=c.HIDDEN_SIZE, bidirectional=c.USE_BIDIRECTIONAL):
		super(LCSP_LSTM_Attn, self).__init__()
		self.lstm = torch.nn.LSTM(input_size=input_size,
		                          hidden_size=hidden_size,
		                          batch_first=True,
		                          bidirectional=bidirectional)
		if bidirectional:
			hidden_size = hidden_size * 2
		self.attn = Attention_Layer(input_size=hidden_size)
		self.clf = Classifier(input_size=hidden_size)

	def forward(self, x):
		o, (h_T, c_T) = self.lstm(x)
		w = self.attn(o)
		h_w = torch.mul(o, w)
		h_w = torch.sum(h_w, dim=1)
		proba = self.clf(h_w)
		return proba, w


''''''''''''''' LSTM + RV-key Embedding '''''''''''''''


class LCSP_LSTM_Emb(torch.nn.Module):
	"""
	LSTM with embedding mechanism for low capacity status prediction problem.
	This model include lstm, embedding layer and classifier module.
	:param input_size: int, dimension of raw - features (exluded time step)
	:param hidden_size: int, dimension of hidden size for lstm module
	:param bidirectional: bool, if True, use bidirectional-lstm
	:param num_embeddings: int, number of unique RV-key (key-index)
	:param embed_dim: int, dimension of output embedding vector
	"""

	def __init__(self, input_size=c.FEATURE_DIM, hidden_size=c.HIDDEN_SIZE, bidirectional=c.USE_BIDIRECTIONAL,
	             num_embeddings=c.NUM_IDS, embed_dim=c.EMBED_DIM):
		super(LCSP_LSTM_Emb, self).__init__()
		self.lstm = torch.nn.LSTM(input_size=input_size,
		                          hidden_size=hidden_size,
		                          batch_first=True,
		                          bidirectional=bidirectional)
		if bidirectional:
			hidden_size = hidden_size * 2
		self.emb = Embedding_Layer(num_embeddings=num_embeddings,
		                           embed_dim=embed_dim,align_dim=hidden_size)
		self.clf = Classifier(input_size=hidden_size)

	def forward(self, x, key):
		o, (h_T, c_T) = self.lstm(x)
		h = torch.cat([h_T[0], h_T[1]], dim=1)
		v_k = self.emb(key)
		x = torch.mul(h, v_k)
		proba = self.clf(x)
		return proba, v_k


''''''''''''''' LSTM + Attention + RV-key Embedding  (proposed model) '''''''''''''''


class LCSP_Net(torch.nn.Module):
	"""
	Proposed model that inclsuion embedding and attention mechanisms.
	:param input_size: int, dimension of raw - features (exluded time step)
	:param hidden_size: int, dimension of hidden size for lstm module
	:param bidirectional: bool, if True, use bidirectional-lstm
	:param num_embeddings: int, number of unique RV-key (key-index)
	:param embed_dim: int, dimension of output embedding vector
	"""

	def __init__(self, input_size=c.FEATURE_DIM, hidden_size=c.HIDDEN_SIZE, bidirectional=c.USE_BIDIRECTIONAL,
	             num_embeddings=c.NUM_IDS, embed_dim=c.EMBED_DIM):
		super(LCSP_Net, self).__init__()
		self.lstm = torch.nn.LSTM(input_size=input_size,
		                          hidden_size=hidden_size,
		                          batch_first=True,
		                          bidirectional=bidirectional)
		if bidirectional:
			hidden_size = hidden_size * 2
		self.attn = Attention_Layer(input_size=hidden_size)
		self.emb = Embedding_Layer(num_embeddings=num_embeddings,
								   embed_dim=embed_dim, align_dim=hidden_size)
		self.clf = Classifier(input_size=hidden_size)

	def forward(self, x, key):
		o, (h_T, c_T) = self.lstm(x)
		w = self.attn(o)
		h_w = torch.mul(o, w)
		h_w = torch.sum(h_w, dim=1)
		v_k = self.emb(key)
		x = torch.mul(h_w, v_k)
		proba = self.clf(x)
		return proba, v_k, w
