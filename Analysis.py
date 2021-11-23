import os
import operator
import numpy as np
import matplotlib.pyplot as plt
import math

# from util.text_util import word_sentence_length

# class statistics():
# 	def __init__(self, category, text_type):

# 		self.category = category
# 		self.text_type = text_type

# 		self.project_sentences_list =[]
# 		self.class_list = []

# 		self.total_project_count = 0
# 		self.success_count = 0
# 		self.fail_count = 0
# 		self.Total_avg_number_of_sentence = 0
# 		self.Total_avg_number_of_words_in_sentence = 0
# 		self.success_avg_number_of_sentence = 0
# 		self.success_avg_number_of_words_in_sentence = 0
# 		self.fail_avg_number_of_sentence = 0
# 		self.fail_avg_number_of_words_in_sentence = 0

# 		if os.path.isdir("./statistics_result") == False:
# 			os.mkdir("./statistics_result")
# 		if os.path.isdir("./statistics_result/detail") == False:
# 			os.mkdir("./statistics_result/detail")
# 		if os.path.isdir("./attention_result") == False:
# 			os.mkdir("./attention_result")

# 	def word_sentence_distribution(self, train_x, train_y, test_x, test_y):

# 		for project_sentences in train_x:
# 			project_sentences = ' '.join(project_sentences)	
# 			self.project_sentences_list.append(project_sentences)
				
# 		for project_sentences in test_x:
# 			project_sentences = ' '.join(project_sentences)
# 			self.project_sentences_list.append(project_sentences)

# 		class_status1 = list(train_y.argmax(axis=1))
# 		class_status2 = list(test_y.argmax(axis=1))
# 		self.class_list = class_status1 + class_status2


# 		for project, status in zip(self.project_sentences_list, self.class_list):
# 			self.total_project_count += 1
# 			(number_of_sentence, avg_number_of_words_in_sentence, number_of_words_in_sentence_list) = word_sentence_length(project)

# 			self.Total_avg_number_of_sentence += number_of_sentence
# 			self.Total_avg_number_of_words_in_sentence += avg_number_of_words_in_sentence
			
# 			if status == 1:
# 				self.success_count += 1
# 				self.success_avg_number_of_sentence += number_of_sentence
# 				self.success_avg_number_of_words_in_sentence += avg_number_of_words_in_sentence

# 				f = open("./statistics_result/detail/" + self.category + "_" + self.text_type + "_success.txt", "a")
# 				number_of_words_in_sentence = ' '.join(map(str, number_of_words_in_sentence_list))
# 				f.write(str(number_of_sentence) + "\t" +str(avg_number_of_words_in_sentence)+"\t"+str(number_of_words_in_sentence)+"\n")
# 				f.close()

# 			elif status == 0:
# 				self.fail_count += 1
# 				self.fail_avg_number_of_sentence += number_of_sentence
# 				self.fail_avg_number_of_words_in_sentence += avg_number_of_words_in_sentence

# 				f = open("./statistics_result/detail/" + self.category + "_" + self.text_type + "_fail.txt", "a")
# 				number_of_words_in_sentence = ' '.join(map(str, number_of_words_in_sentence_list))
# 				f.write(str(number_of_sentence) + "\t" +str(avg_number_of_words_in_sentence)+"\t"+str(number_of_words_in_sentence)+"\n")
# 				f.close()

# 			f = open("./statistics_result/detail/" + self.category + "_" + self.text_type + ".txt", "a")
# 			number_of_words_in_sentence = ' '.join(map(str, number_of_words_in_sentence_list))
# 			f.write(str(number_of_sentence) + "\t" +str(avg_number_of_words_in_sentence)+"\t"+str(number_of_words_in_sentence)+"\n")
# 			f.close()


# 		f = open("./statistics_result/" + self.category + "_" + self.text_type + ".txt", "w")
# 		f.write("status\tavg_sentence\tavg_words_in_sentence\n")
# 		f.write("success\t%0.2f\t%0.2f\n"  % ((self.success_avg_number_of_sentence / self.success_count), (self.success_avg_number_of_words_in_sentence / self.success_count)))
# 		f.write("fail\t%0.2f\t%0.2f\n"  % ((self.fail_avg_number_of_sentence / self.fail_count), (self.fail_avg_number_of_words_in_sentence / self.fail_count)))
# 		f.write("total\t%0.2f\t%0.2f\n"  % ((self.Total_avg_number_of_sentence / self.total_project_count), (self.Total_avg_number_of_words_in_sentence / self.total_project_count)))
# 		f.close()




class distribution():
	def __init__(self, data_type, category, text_type):
		self.category = category
		self.text_type = text_type
		self.data_type = data_type
		self.top_word = 100

	def weight_distribution(self):

		if os.path.isdir("./distribution_result") == False:
			os.mkdir("./distribution_result")
		if os.path.isdir("./distribution_result/" + self.data_type + "_" + self.category + "_" + self.text_type) == False:
			os.mkdir("./distribution_result/" + self.data_type + "_" + self.category + "_" + self.text_type)

		pred_success_weight_word_distribution = dict()
		pred_fail_weight_word_distribution = dict()
		real_success_weight_word_distribution = dict()
		real_fail_weight_word_distribution = dict()
		full_weight_word_distribution = dict()
		calculate_full_weight_word_distribution = dict()

		pred_success_weight_length = dict()
		pred_fail_weight_length = dict()
		real_success_weight_length = dict()
		real_fail_weight_length = dict()
		full_weight_length = dict()
		calculate_full_weight_length = dict()

		class_results = open("./attention_result/" + self.data_type + "_" + self.category + "_" + self.text_type + "_classification_result.txt", "r").readlines()

		for result in class_results:

			result = result.strip().split("\t")
			if result[0] == "num":
				continue

			project_num = int(result[0])
			real_class = int(result[1])
			pred_class = int(result[2])
			probability_success = float(result[3])
			probability_fail = float(result[4])

			lines = open("./attention_result/" + self.data_type + "_" + self.category + "_" + self.text_type + "/" +str(real_class) + "/" + str(project_num) + ".txt", "r").readlines()
			
			for line in lines:
				line = line.split("\t")
				word_weight_list = line[1:-1]

				number_of_word = len(word_weight_list)
				
				if number_of_word != 0:
					avg_weight = 1.0 / number_of_word

					for word in word_weight_list:
						word_list = word.split(" ")
						word = word_list[0]
						word_weight = word_list[1]

						calculate_word_weight = float(word_weight) - avg_weight

						####Full class
						if word in full_weight_word_distribution.keys():
							full_weight_word_distribution[word].append(calculate_word_weight)
						else:
							full_weight_word_distribution[word] = []
							full_weight_word_distribution[word].append(calculate_word_weight)

						####classify as real class
						if real_class == 1: ##success
							if word in real_success_weight_word_distribution.keys():
								real_success_weight_word_distribution[word].append(calculate_word_weight)
							else:
								real_success_weight_word_distribution[word] = []
								real_success_weight_word_distribution[word].append(calculate_word_weight)

						elif real_class == 0: ##fail
							if word in real_fail_weight_word_distribution.keys():
								real_fail_weight_word_distribution[word].append(calculate_word_weight)
							else:
								real_fail_weight_word_distribution[word] = []
								real_fail_weight_word_distribution[word].append(calculate_word_weight)

						####classify as pred class
						if (pred_class == 1) and (real_class ==1): ##success
							if word in pred_success_weight_word_distribution.keys():
								pred_success_weight_word_distribution[word].append(calculate_word_weight)
							else:
								pred_success_weight_word_distribution[word] = []
								pred_success_weight_word_distribution[word].append(calculate_word_weight)

						elif (pred_class == 0) and (real_class ==0): ##fail
							if word in pred_fail_weight_word_distribution.keys():
								pred_fail_weight_word_distribution[word].append(calculate_word_weight)
							else:
								pred_fail_weight_word_distribution[word] = []
								pred_fail_weight_word_distribution[word].append(calculate_word_weight)
	

		####Full class
		for num, word in enumerate(full_weight_word_distribution):
			full_weight_length[word] = len(full_weight_word_distribution[word])

		full_weight_length = sorted(full_weight_length.items(), key=operator.itemgetter(1), reverse=True)
		full_weight_top = full_weight_length[:self.top_word]

		####classify as real class
		for num, word in enumerate(real_success_weight_word_distribution):
			real_success_weight_length[word] = len(real_success_weight_word_distribution[word])
		for num, word in enumerate(real_fail_weight_word_distribution):
			real_fail_weight_length[word] = len(real_fail_weight_word_distribution[word])

		real_success_weight_length = sorted(real_success_weight_length.items(), key=operator.itemgetter(1), reverse=True)
		real_fail_weight_length = sorted(real_fail_weight_length.items(), key=operator.itemgetter(1), reverse=True)
		real_success_weight_top = real_success_weight_length[:self.top_word]
		real_fail_weight_top = real_fail_weight_length[:self.top_word]

		####classify as pred class
		for num, word in enumerate(pred_success_weight_word_distribution):
			pred_success_weight_length[word] = len(pred_success_weight_word_distribution[word])
		for num, word in enumerate(pred_fail_weight_word_distribution):
			pred_fail_weight_length[word] = len(pred_fail_weight_word_distribution[word])

		pred_success_weight_length = sorted(pred_success_weight_length.items(), key=operator.itemgetter(1), reverse=True)
		pred_fail_weight_length = sorted(pred_fail_weight_length.items(), key=operator.itemgetter(1), reverse=True)
		pred_success_weight_top = pred_success_weight_length[:self.top_word]
		pred_fail_weight_top = pred_fail_weight_length[:self.top_word]


		sum_weight_list = dict()
		sum_weight_list_divided_numofword = dict()
		positive_variation_list = dict()
		positive_variation_divided_numofword = dict()
		pos_neg_count_weight_list = dict()
		
		for word_and_weight in full_weight_length:
			word = word_and_weight[0]
			full_weight_list = full_weight_word_distribution[word]
			
			Total_weight = 0.0
			positive_variation = 0
			pos_count = 0
			neg_count = 0
			for weight in full_weight_list:
				Total_weight += weight

				if weight > 0:
					pos_count +=1
					positive_variation += weight
				elif weight <0:
					neg_count -=1

			sum_weight_list[word] = Total_weight
			sum_weight_list_divided_numofword[word] = Total_weight / float(len(full_weight_list))
			if pos_count != 0:
				positive_variation_list[word] = positive_variation
				positive_variation_divided_numofword[word] = positive_variation / float(pos_count)
			pos_neg_count_weight_list[word] = [(pos_count + neg_count), pos_count, neg_count]


		sum_weight_list = sorted(sum_weight_list.items(), key=operator.itemgetter(1), reverse=True)
		# sum_weight_list_divided_numofword = sorted(sum_weight_list_divided_numofword.items(), key=operator.itemgetter(1), reverse=True)
		# positive_variation_list = sorted(positive_variation_list.items(), key=operator.itemgetter(1), reverse=True)
		# positive_variation_divided_numofword = sorted(positive_variation_divided_numofword.items(), key=operator.itemgetter(1), reverse=True)
		
		f = open("./distribution_result/" + self.data_type + "_" + self.category + "_" + self.text_type + "_sum_weight.txt", "w")
		for word, value in sum_weight_list:
			weightcount = pos_neg_count_weight_list[word]
			f.write(str(weightcount[0])+"\t" +str(weightcount[1])+"\t" +str(weightcount[2])+ "\t" + str(word) + "\t" + str(value) + "\t")

			if word in sum_weight_list_divided_numofword.keys():
				f.write(str(sum_weight_list_divided_numofword[word]) + "\t")
			else:
				f.write(str(0) + "\t")

			if word in positive_variation_list.keys():
				f.write(str(positive_variation_list[word]) + "\t")
			else:
				f.write(str(0) + "\t")

			if word in positive_variation_divided_numofword.keys():
				f.write(str(positive_variation_divided_numofword[word]) + "\n")
			else:
				f.write(str(0) + "\n")
		f.close()


		# 	if word not in pred_success_weight_word_distribution:
		# 		success_weight_list = [0]
		# 	else:
		# 		success_weight_list = pred_success_weight_word_distribution[word]
		# 	success_weight_list = [float(weight) for weight in success_weight_list]
		# 	success_weight_arr = np.asarray(success_weight_list)

		# 	if word not in pred_fail_weight_word_distribution:
		# 		fail_weight_list = [0]
		# 	else:
		# 		fail_weight_list = pred_fail_weight_word_distribution[word]
		# 	fail_weight_list = [float(weight) for weight in fail_weight_list]
		# 	fail_weight_arr = np.asarray(fail_weight_list)

		# 	x = np.arange(-0.5, 0.5, 0.01)

		# 	success_hist, bins = np.histogram(success_weight_arr, bins=x)
		# 	fail_hist, bins = np.histogram(fail_weight_arr, bins=x)

		# 	success_hist = np.append(success_hist, 0)
		# 	fail_hist = np.append(fail_hist, 0)

		# 	fig, ax = plt.subplots(figsize=(10,10))
		# 	# line1 = ax.plot(x, success_hist, color='b', label="success", alpha=0.5)
		# 	# line2 = ax.plot(x, fail_hist, color='r', label="fail", alpha=0.5)

		# 	lin1 = ax.hist(success_weight_arr, bins=x, color='b', label="success", alpha=0.5)
		# 	ax.grid()
		# 	ax.legend()
		# 	ax.set_title(word, fontsize=30)
		# 	ax.set_ylabel("word_count")
		# 	ax.set_xlabel("weight")

		# 	word_count = len(full_weight_word_distribution[word])
		# 	fig.savefig("./distribution_result/" + self.data_type + "_" + self.category + "_" + self.text_type + "/"+ str(word_count) + "_" + str(word) + ".png")
		# 	plt.close()

