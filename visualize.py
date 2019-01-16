# coding=utf-8

import os
import sys
import logging
from sklearn.externals import joblib
import numpy as np
import torch
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec,ticker,rcParams

from config import config as c

import warnings
warnings.filterwarnings('ignore')

colornames = {'aliceblue': '#F0F8FF', 'antiquewhite': '#FAEBD7', 'aqua': '#00FFFF', 'aquamarine': '#7FFFD4', 'azure': '#F0FFFF', 'beige': '#F5F5DC', 'bisque': '#FFE4C4', 'black': '#000000', 'blanchedalmond': '#FFEBCD', 'blue': '#0000FF', 'blueviolet': '#8A2BE2', 'brown': '#A52A2A', 'burlywood': '#DEB887', 'cadetblue': '#5F9EA0', 'chartreuse': '#7FFF00', 'chocolate': '#D2691E', 'coral': '#FF7F50', 'cornflowerblue': '#6495ED', 'cornsilk': '#FFF8DC', 'crimson': '#DC143C', 'cyan': '#00FFFF', 'darkblue': '#00008B', 'darkcyan': '#008B8B', 'darkgoldenrod': '#B8860B', 'darkgray': '#A9A9A9', 'darkgreen': '#006400', 'darkkhaki': '#BDB76B', 'darkmagenta': '#8B008B', 'darkolivegreen': '#556B2F', 'darkorange': '#FF8C00', 'darkorchid': '#9932CC', 'darkred': '#8B0000', 'darksalmon': '#E9967A', 'darkseagreen': '#8FBC8F', 'darkslateblue': '#483D8B', 'darkslategray': '#2F4F4F', 'darkturquoise': '#00CED1', 'darkviolet': '#9400D3', 'deeppink': '#FF1493', 'deepskyblue': '#00BFFF', 'dimgray': '#696969', 'dodgerblue': '#1E90FF', 'firebrick': '#B22222', 'floralwhite': '#FFFAF0', 'forestgreen': '#228B22', 'fuchsia': '#FF00FF', 'gainsboro': '#DCDCDC', 'ghostwhite': '#F8F8FF', 'gold': '#FFD700', 'goldenrod': '#DAA520', 'gray': '#808080', 'green': '#008000', 'greenyellow': '#ADFF2F', 'honeydew': '#F0FFF0', 'hotpink': '#FF69B4', 'indianred': '#CD5C5C', 'indigo': '#4B0082', 'ivory': '#FFFFF0', 'khaki': '#F0E68C', 'lavender': '#E6E6FA', 'lavenderblush': '#FFF0F5', 'lawngreen': '#7CFC00', 'lemonchiffon': '#FFFACD', 'lightblue': '#ADD8E6', 'lightcoral': '#F08080', 'lightcyan': '#E0FFFF', 'lightgoldenrodyellow': '#FAFAD2', 'lightgreen': '#90EE90', 'lightgray': '#D3D3D3', 'lightpink': '#FFB6C1', 'lightsalmon': '#FFA07A', 'lightseagreen': '#20B2AA', 'lightskyblue': '#87CEFA', 'lightslategray': '#778899', 'lightsteelblue': '#B0C4DE', 'lightyellow': '#FFFFE0', 'lime': '#00FF00', 'limegreen': '#32CD32', 'linen': '#FAF0E6', 'magenta': '#FF00FF', 'maroon': '#800000', 'mediumaquamarine': '#66CDAA', 'mediumblue': '#0000CD', 'mediumorchid': '#BA55D3', 'mediumpurple': '#9370DB', 'mediumseagreen': '#3CB371', 'mediumslateblue': '#7B68EE', 'mediumspringgreen': '#00FA9A', 'mediumturquoise': '#48D1CC', 'mediumvioletred': '#C71585', 'midnightblue': '#191970', 'mintcream': '#F5FFFA', 'mistyrose': '#FFE4E1', 'moccasin': '#FFE4B5', 'navajowhite': '#FFDEAD', 'navy': '#000080', 'oldlace': '#FDF5E6', 'olive': '#808000', 'olivedrab': '#6B8E23', 'orange': '#FFA500', 'orangered': '#FF4500', 'orchid': '#DA70D6', 'palegoldenrod': '#EEE8AA', 'palegreen': '#98FB98', 'paleturquoise': '#AFEEEE', 'palevioletred': '#DB7093', 'papayawhip': '#FFEFD5', 'peachpuff': '#FFDAB9', 'peru': '#CD853F', 'pink': '#FFC0CB', 'plum': '#DDA0DD', 'powderblue': '#B0E0E6', 'purple': '#800080', 'red': '#FF0000', 'rosybrown': '#BC8F8F', 'royalblue': '#4169E1', 'saddlebrown': '#8B4513', 'salmon': '#FA8072', 'sandybrown': '#FAA460', 'seagreen': '#2E8B57', 'seashell': '#FFF5EE', 'sienna': '#A0522D', 'silver': '#C0C0C0', 'skyblue': '#87CEEB', 'slateblue': '#6A5ACD', 'slategray': '#708090', 'snow': '#FFFAFA', 'springgreen': '#00FF7F', 'steelblue': '#4682B4', 'tan': '#D2B48C', 'teal': '#008080', 'thistle': '#D8BFD8', 'tomato': '#FF6347', 'turquoise': '#40E0D0', 'violet': '#EE82EE', 'wheat': '#F5DEB3', 'white': '#FFFFFF', 'whitesmoke': '#F5F5F5', 'yellow': '#FFFF00', 'yellowgreen': '#9ACD32'}

# init device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

''''''''''''''' Get data and predict  '''''''''''''''

def get_visualize_data(data_path,key2ids_dict):
	"""
	Run inference to get output of the model .
	:param data_path: string, path for data that used for visualizing
	:param key2ids_dict: dict, convert RV-key(string) to index (int). e.g. {'Dsv3':1,...}
	"""
	data = joblib.load(data_path)
	x = data['data_x']
	key = data['data_key']
	key_ids = [key2ids_dict.get(r, 0) for r in key]
	date = data['Date']
	print(date)
	print(x.shape, len(key))
	return [x,key, key_ids]

def inference_for_visualzie(data, model_path):
	"""
	Run inference to get output of the model .
	:param data: numpy array, input data for model
	:param model_path: string, path for model that used for inference
	"""
	model = torch.load(model_path)
	[x, key, key_ids] = data
	x = torch.tensor(x, dtype=torch.float).to(device)
	key_ids = torch.tensor(key_ids, dtype=torch.long).to(device)
	output, v_k, w = model(x, key_ids)
	output= output.cpu().detach().numpy()
	v_k = v_k.cpu().detach().numpy()
	w = w.cpu().detach().numpy()
	print('output shape:{0}, key vector shape:{1}, attention weight shape:{2}'.format(output.shape, v_k.shape, w.shape))
	return output,v_k, w

''''''''''''''' Visualizing  '''''''''''''''

def Visualize_EmbeddingVector(v_k,key):
	"""
	Visualizing the embedding key vector by t-SNE .
	:param v_k: numpy array, RV-key vector
	:param key: array or list like, string format RV-keys
	"""
	sns.set(style="darkgrid", font_scale=2.5)
	rcParams.update({'font.size': 35})
	x_tsne = TSNE(n_components=2, angle=0.2, random_state=520).fit_transform(np.array(v_k))
	print('using t-sne to decrease dimention:{0}'.format(x_tsne.shape))
	data_keys = np.array(key)
	plt.figure(figsize=(25, 25))
	for ii, kk in enumerate(sorted(list(set(data_keys)))): # you can only visualize some keys
		inds = []
		for jj in range(len(data_keys)):
			if data_keys[jj] == kk:
				inds.append(jj)
				break
		s_tsne = x_tsne[inds]  # just visualize unique keys
		x, y = s_tsne[:, 0], s_tsne[:, 1]
		plt.scatter(x, y, color=list(colornames.keys())[ii + 10])
		plt.text(x, y, kk,color=list(colornames.keys())[ii + 10])
	title = 'Visualization of RV-key vectors using t-SNE'
	plt.title(title)
	plt.show()

def Visualize_AttentionWeight(w,x,index=0):
	"""
	Visualizing the Attention weight by heat map .
	:param w: numpy array, attention weight
	:param x: numpy array, time-series inpur feature for model
	:param index: int, index of visualizing samples
	"""
	ww=np.swapaxes(w[index],0,1)
	raw_data=np.swapaxes(x[index],0,1)
	feature_ind = 11# you can plot one or more features. In there, we just plot one features which index=11
	plt.figure(figsize=(15, 10))
	# gs = gridspec.GridSpec(2, 1,height_ratios=[3,1])
	gs = gridspec.GridSpec(7, 37)
	ax1 = plt.subplot(gs[:4, :29])
	ax1.plot(raw_data[feature_ind], label='(feature name)', linewidth=5.0)
	ax1.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])  # In our task, time step=14
	ax1.set_xticklabels([14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
	ax1.set_yticks([])
	ax1.set_xlabel("Δt", fontsize=30)
	ax1.legend()
	ax2 = plt.subplot(gs[5:, :-1])
	ax2 = sns.heatmap(ww, cmap="Reds", center=0.06, ax=ax2)  # "YlGnBu""Reds""sns.cubehelix_palette(8)
	ax2.set_xticklabels([14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
	ax2.set_yticks([])
	ax2.set_xlabel("Δt", fontsize=30)
	plt.tight_layout()
	plt.show()

''''''''''''''' main '''''''''''''''

def main():
	key2ids_dict=joblib.load(c.key2ids_path)
	data = get_visualize_data(data_path=c.TEST_DATA_PATH,key2ids_dict=key2ids_dict)
	[x,  key, key_ids]=data
	output, v_k, w = inference_for_visualzie(data=data, model_path=c.BEST_MODEL_PATH)
	Visualize_EmbeddingVector(v_k, key)
	Visualize_AttentionWeight(w, x, index=0)

if __name__ == '__main__':
	main()