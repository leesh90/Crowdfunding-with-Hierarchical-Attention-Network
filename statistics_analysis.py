import os

from util.text_util import word_sentence_length

class statistics():
	def __init__(self, category, text_type):

		self.category = category
		self.text_type = text_type

		self.project_sentences_list =[]
		self.class_list = []

		self.total_project_count = 0
		self.success_count = 0
		self.fail_count = 0
		self.Total_avg_number_of_sentence = 0
		self.Total_avg_number_of_words_in_sentence = 0
		self.success_avg_number_of_sentence = 0
		self.success_avg_number_of_words_in_sentence = 0
		self.fail_avg_number_of_sentence = 0
		self.fail_avg_number_of_words_in_sentence = 0

		if os.path.isdir("./statistics_result") == False:
			os.mkdir("./statistics_result")
		if os.path.isdir("./statistics_result/detail") == False:
			os.mkdir("./statistics_result/detail")
		if os.path.isdir("./attention_result") == False:
			os.mkdir("./attention_result")

	def word_sentence_distribution(self, train_x, train_y, test_x, test_y):

		for project_sentences in train_x:
			project_sentences = ' '.join(project_sentences)	
			self.project_sentences_list.append(project_sentences)
				
		for project_sentences in test_x:
			project_sentences = ' '.join(project_sentences)
			self.project_sentences_list.append(project_sentences)

		class_status1 = list(train_y.argmax(axis=1))
		class_status2 = list(test_y.argmax(axis=1))
		self.class_list = class_status1 + class_status2


		for project, status in zip(self.project_sentences_list, self.class_list):
			self.total_project_count += 1
			(number_of_sentence, avg_number_of_words_in_sentence, number_of_words_in_sentence_list) = word_sentence_length(project)

			self.Total_avg_number_of_sentence += number_of_sentence
			self.Total_avg_number_of_words_in_sentence += avg_number_of_words_in_sentence
			
			if status == 1:
				self.success_count += 1
				self.success_avg_number_of_sentence += number_of_sentence
				self.success_avg_number_of_words_in_sentence += avg_number_of_words_in_sentence

				f = open("./statistics_result/detail/" + self.category + "_" + self.text_type + "_success.txt", "a")
				number_of_words_in_sentence = ' '.join(map(str, number_of_words_in_sentence_list))
				f.write(str(number_of_sentence) + "\t" +str(avg_number_of_words_in_sentence)+"\t"+str(number_of_words_in_sentence)+"\n")
				f.close()

			elif status == 0:
				self.fail_count += 1
				self.fail_avg_number_of_sentence += number_of_sentence
				self.fail_avg_number_of_words_in_sentence += avg_number_of_words_in_sentence

				f = open("./statistics_result/detail/" + self.category + "_" + self.text_type + "_fail.txt", "a")
				number_of_words_in_sentence = ' '.join(map(str, number_of_words_in_sentence_list))
				f.write(str(number_of_sentence) + "\t" +str(avg_number_of_words_in_sentence)+"\t"+str(number_of_words_in_sentence)+"\n")
				f.close()

			f = open("./statistics_result/detail/" + self.category + "_" + self.text_type + ".txt", "a")
			number_of_words_in_sentence = ' '.join(map(str, number_of_words_in_sentence_list))
			f.write(str(number_of_sentence) + "\t" +str(avg_number_of_words_in_sentence)+"\t"+str(number_of_words_in_sentence)+"\n")
			f.close()


		f = open("./statistics_result/" + self.category + "_" + self.text_type + ".txt", "w")
		f.write("status\tavg_sentence\tavg_words_in_sentence\n")
		f.write("success\t%0.2f\t%0.2f\n"  % ((self.success_avg_number_of_sentence / self.success_count), (self.success_avg_number_of_words_in_sentence / self.success_count)))
		f.write("fail\t%0.2f\t%0.2f\n"  % ((self.fail_avg_number_of_sentence / self.fail_count), (self.fail_avg_number_of_words_in_sentence / self.fail_count)))
		f.write("total\t%0.2f\t%0.2f\n"  % ((self.Total_avg_number_of_sentence / self.total_project_count), (self.Total_avg_number_of_words_in_sentence / self.total_project_count)))
		f.close()




	def attention_distribution(self):

		sentence_weight_list = []
		word_weight_list = []




