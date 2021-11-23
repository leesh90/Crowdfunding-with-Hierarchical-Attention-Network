from HAN import Hierarchical_attention_networks
from Analysis import *
from word2vec_visual import word_vector_visualization
import sys

if __name__ == '__main__':

	data_type = sys.argv[1]
	category = sys.argv[2]#, "Art", "design", "game"]
	text_type = sys.argv[3]#, "update","b_comment", "c_comment", "full_comment"]

	good_num_fold = sys.argv[4]

	# sys.stdout = open("./log_dir/" + data_type +"_" + category + "_" + text_type + ".txt", 'w')


	TRAINING_SPLIT = 0.8
	VALIDATION_SPLIT = 0.1
	TEST_SPLIT = 0.1

	MAX_SEQUENCE_LENGTH = 30
	MAX_SENTS = 100

	h = Hierarchical_attention_networks(TRAINING_SPLIT=TRAINING_SPLIT, VALIDATION_SPLIT=VALIDATION_SPLIT, 
		TEST_SPLIT=TEST_SPLIT, MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH, MAX_SENTS=MAX_SENTS, K_num = 10)


	#data read
	crowdfunding_data = h.data_read(data_type, category, text_type)
	print ("-----read completed-----")

	# pre-processing data
	data, labels, texts = h.text_preprocessing(crowdfunding_data)
	print ("-----preprocessing completed-----")

	# Loading pre-trained word embedding
	h.make_embedding_layer()
	print ("-----embedding completed-----")

	# split dataset
	# (train_x, train_y), (valid_x, valid_y), (test_x ,test_y) = h.data_split(data, labels)
	(x, y, ori_y, kfold) = h.data_split(data, labels)
	print ("-----data split completed-----")


	acc_per_fold = []

	num_fold = 1
	for train, test in kfold.split(x, ori_y):

		train_skip, test_skip = h.fold_position_save_load(num_fold, 1, train, test)
 
		# model compile
		h.HAN_layer()
		
		#Training
		hist = h.training(x, y, train, num_fold)

		#Training_result
		acc_per_fold = h.train_progress(x, y, test, hist, num_fold, acc_per_fold)

		if num_fold == 1:
			break
		num_fold += 1

	# Loading weights
	# num_fold = int(good_num_fold)
	# h.load_weights(num_fold)
	# print ("-----weight load completed-----")


	# # attention word distribution
	# for num_fold in [1,2,3,4,5]:
	# 	if num_fold == int(good_num_fold):

	# 		train, test = h.fold_position_save_load(num_fold, 2)
	# 		test_texts = [texts[index] for index in test]
	# 		h.activation_map_distribution(x[test], y[test], test_texts)
	# 		# h.activation_visualization(x[test], y[test], test_texts)
	# print ("-----word distribution completed-----")

	# # Attention visualization
	# dis = distribution(data_type = data_type, category = category, text_type = text_type)
	# dis.weight_distribution()


	# # word vector visaulization
	# vis = word_vector_visualization(data_type = data_type, category = category, text_type = text_type)
	# vis.load_data()
	# vis.tsne()
	# vis.plot_with_labels()

	h.clear_model()
