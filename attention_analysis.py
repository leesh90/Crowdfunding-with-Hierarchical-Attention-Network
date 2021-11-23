import numpy as np
import os
from util.text_util import normalize
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

class attention_analysis():
	def __init__(self, category, text_type):
		self.category = category
		self.text_type = text_type

		if os.path.isdir("./attention_result") == False:
			os.mkdir("./attention_result")

	def attention_distribution(self):

		sentence_weight_list = []
		word_weight_list = []

		project_file_list = os.listdir("./attention_result/" + self.category + "_" + self.text_type  + "_train/")
		for file in project_file_list:
			f = open("./attention_result/" + self.category + "_" + self.text_type  + "_train/" + file, "r")
			lines = f.readlines()
			for line in lines:
				line = line.strip().split("\t")

				sentence_att_weight = float(line[0])
				sentence_att_weight = round(sentence_att_weight, 2)
				sentence_weight_list.append(sentence_att_weight)

				for num in range(1, len(line)):
					word = line[num].split(" ")
					word_weight = float(word[1]) * sentence_att_weight
					word_weight = round(word_weight, 2)
					word_weight_list.append(word_weight)


		project_file_list = os.listdir("./attention_result/" + self.category + "_" + self.text_type  + "_test/")
		for file in project_file_list:
			f = open("./attention_result/" + self.category + "_" + self.text_type  + "_test/" + file, "r")
			lines = f.readlines()
			for line in lines:
				line = line.strip().split("\t")

				sentence_att_weight = float(line[0])
				sentence_att_weight = round(sentence_att_weight, 2)
				sentence_weight_list.append(sentence_att_weight)

				for num in range(1, len(line)):
					word = line[num].split(" ")
					word_weight = float(word[1]) * sentence_att_weight
					word_weight = round(word_weight, 2)
					word_weight_list.append(word_weight)



		fig, axs = plt.subplots(ncols=2, nrows=2)
		fig.set_size_inches(16, 16)

		sns.distplot(sentence_weight_list, ax=axs[0, 0])
		axs[0, 0].set_title("sentence_weight_distribution(" + str(len(sentence_weight_list)) + ")")
		axs[0, 0].set_xlabel("Weight(0~1)")
		axs[0, 0].set_ylabel("sentence_count")
		axs[0, 0].set_xticks([0,1,0.1])
		axs[0, 0].grid()

		sns.distplot(sentence_weight_list, hist_kws=dict(cumulative=True), kde_kws=dict(cumulative=True), ax=axs[0, 1])
		axs[0, 1].set_title("sentence_weight_distribution(CDF)")
		axs[0, 1].set_xlabel("Weight(0~1)")
		axs[0, 1].set_ylabel("CDF")
		axs[0, 1].set_xticks([0,1,0.1])
		axs[0, 1].grid()

		# sns.distplot(word_weight_list, ax=axs[1, 0])
		# axs[1, 0].set_title("word_weight_distribution(" + str(len(word_weight_list)) + ")")
		# axs[1, 0].set_xlabel("Weight(0~1)")
		# axs[1, 0].set_ylabel("word_count")
		# axs[1, 0].set_xticks([0,1,0.1])
		# axs[1, 0].grid()

		# sns.distplot(word_weight_list, hist_kws=dict(cumulative=True), kde_kws=dict(cumulative=True), ax=axs[1, 1])
		# axs[1, 1].set_title("word_weight_distribution(CDF)")
		# axs[1, 1].set_xlabel("Weight(0~1)")
		# axs[1, 1].set_ylabel("CDF")
		# axs[1, 1].set_xticks([0,1,0.1])
		# axs[1, 1].grid()

		fig.savefig("./attention_result/" + self.category + "_" + self.text_type + "_distribution.png")


