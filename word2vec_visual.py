from gensim.models import KeyedVectors
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import csv


class word_vector_visualization:
	def __init__(self, data_type, category, text_type):
		self.category = category
		self.text_type = text_type
		self.data_type = data_type


		self.embedding = np.array([])
		self.decresed_dim_embedding = []
		self.word_model = None
		self.word_list = []
		self.point_list_1 = []
		self.point_list_2 = []

	def load_vector(self):
		self.word_model = KeyedVectors.load_word2vec_format('./saved_models/crawl-300d-2M.vec')

	def load_data(self):

		word_list = []
		point_list_1 = []
		point_list_2 = []

		f = open("./distribution_result/" + self.data_type + "_" + self.category + "_" + self.text_type + "_sum_weight.txt", "r", encoding='utf-8')
		rdr = csv.reader(f)
		for line in rdr:
			line = line[0].split("\t")
			word_list.append(line[3])
			point_list_1.append(float(line[4]))
			point_list_2.append(float(line[5]))
		f.close()

		word_list_tmp = word_list[:100] + word_list[-100:]
		point_list_1_tmp = point_list_1[:100] + point_list_1[-100:]
		point_list_2_tmp = point_list_2[:100] + point_list_2[-100:]

		avail_point_list_1 = []
		avail_point_list_2 = []		
		for i, word in enumerate(word_list_tmp):
			if word in self.word_model:
				self.embedding = np.append(self.embedding, self.word_model[word])
				self.word_list.append(word)
				avail_point_list_1.append(point_list_1_tmp[i])
				avail_point_list_2.append(point_list_2_tmp[i])

		self.embedding = self.embedding.reshape([-1,300])

		self.point_list_1 = avail_point_list_1 / np.linalg.norm(avail_point_list_1)
		self.point_list_2 = avail_point_list_2 / np.linalg.norm(avail_point_list_2)

	def tsne(self):
		tsne = TSNE(perplexity=30.0, n_components=2, init='pca', n_iter=5000)
		self.decresed_dim_embedding = tsne.fit_transform(self.embedding)

	def plot_with_labels(self):

		plt.figure(figsize=(18,18))
		plt.scatter(self.decresed_dim_embedding[:,0], self.decresed_dim_embedding[:,1], c=self.point_list_1)

		x = self.decresed_dim_embedding[:,0]
		y = self.decresed_dim_embedding[:,1]

		for i, txt in enumerate(self.word_list):
			plt.annotate(txt, (x[i], y[i]), textcoords='offset points',xytext=(2,10) ,ha='right')
		plt.colorbar()
		plt.savefig("./word2vec_visualization/" + self.data_type + "_" + self.category + "_" + self.text_type + "_1.png")
		plt.close()


		plt.figure(figsize=(18,18))
		plt.scatter(self.decresed_dim_embedding[:,0], self.decresed_dim_embedding[:,1], c=self.point_list_2)

		x = self.decresed_dim_embedding[:,0]
		y = self.decresed_dim_embedding[:,1]

		for i, txt in enumerate(self.word_list):
			plt.annotate(txt, (x[i], y[i]), textcoords='offset points',xytext=(2,10) ,ha='right')
		plt.colorbar()
		plt.savefig("./word2vec_visualization/" + self.data_type + "_" + self.category + "_" + self.text_type + "_2.png")
		plt.close()

