import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np

class visualization_attention():

	def _visualization(self, activation_maps, number):

		a_maps = activation_maps[0]

		word_list = []
		weight_list = []

		for word, weight in a_maps[0]:
			word_list.append(word)
			weight_list.append(weight)

		word_list = np.array(word_list)

		weight_list = np.array(weight_list)
		weight_list = weight_list.reshape(1, len(weight_list))
		
		fig = plt.figure(figsize=(len(word_list), 5))
		plt.rc('xtick', labelsize=16)
		midpoint = (max(weight_list[:, 0]) - min(weight_list[:, 0])) / 2
		heatmap = sn.heatmap(weight_list, xticklabels = word_list ,yticklabels=False, square=True, linewidths=0.1, 
						cmap='coolwarm', center=midpoint, vmin=0, vmax=1)
		plt.xticks(rotation=45)

		plt.savefig("./visualization_result/technology_story/" + str(number) + ".png")




