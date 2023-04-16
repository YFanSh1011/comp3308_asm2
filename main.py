import math
import sys
from Algorithms.knn import classify_nn as knn
from Algorithms.nb import classify_nb as nb


def calculate_statistics(data):
	data_mean = sum(data) / len(data)
	sum_of_squares = 0
	for num in data:
		sum_of_squares += (num - data_mean) ** 2
	stdev = math.sqrt(sum_of_squares / (len(data) - 1))
	return max(data), min(data), data_mean, stdev


def split_data(filename):
	data_in_folds = []
	with open(filename, 'r') as f:
		file_content = f.read().split('fold')
		data_in_folds = []
		for fold in file_content:
			# The first line is the fold number
			# The last line is empty
			entries = fold.splitlines()[1:-1]
			if (len(entries) > 0):
				data_in_folds.append(entries)

		# convert data to float if possible
		convert = lambda a: float(a) if a.strip().replace('.','').isnumeric() else a
		cleaned_data = []
		for fold in data_in_folds:
			cleaned_fold = []
			for entry in fold:
				cleaned_fold.append([convert(x) for x in entry.split(',') if x != ''])
			cleaned_data.append(cleaned_fold)
		return cleaned_data
	

def train_test_split(data, fold_number):
	train = []
	test = []
	for i in range(len(data)):
		if i == fold_number:
			test = data[i]
		else:
			train.extend(data[i])
	return train, test


def ten_fold_cross_validation(data, algorithm, **kwargs):
	accuracies = []
	for i in range(len(data)):
		train, test = train_test_split(data, i)
		if algorithm == nb:
			accuracies.append(algorithm(train, test))
		else:
			accuracies.append(algorithm(train, test, kwargs['k']))
	return accuracies


if __name__ == '__main__':
	data_without_feature_selection = split_data('Data/pima-10-folds.csv')
	data_with_feature_selection = split_data('Data/pima-CFS-folds.csv')
	
	with open('results.txt', 'w') as sys.stdout:
		# 1NN without Feature Selection
		one_nn_accuracies_without_feature_selection = \
			ten_fold_cross_validation(data_without_feature_selection, knn, k=1)
		max_val, min_val, mean_val, stdev = calculate_statistics(one_nn_accuracies_without_feature_selection)
		print("1NN without Feature Selection")
		print('Accuracies: ', one_nn_accuracies_without_feature_selection)
		print('Max: ', max_val)
		print('Min: ', min_val)
		print('Mean: ', mean_val)
		print('Standard Deviation: ', stdev)
		print()

		# 1NN with Feature Selection
		one_nn_accuracies_with_feature_selection = \
			ten_fold_cross_validation(data_with_feature_selection, knn, k=1)
		max_val, min_val, mean_val, stdev = calculate_statistics(one_nn_accuracies_with_feature_selection)
		print("1NN with Feature Selection")
		print('Accuracies: ', one_nn_accuracies_with_feature_selection)
		print('Max: ', max_val)
		print('Min: ', min_val)
		print('Mean: ', mean_val)
		print('Standard Deviation: ', stdev)
		print()

		# 5NN without Feature Selection
		five_nn_accuracies_without_feature_selection = \
			ten_fold_cross_validation(data_without_feature_selection, knn, k=5)
		max_val, min_val, mean_val, stdev = calculate_statistics(five_nn_accuracies_without_feature_selection)
		print("5NN without Feature Selection")
		print('Accuracies: ', five_nn_accuracies_without_feature_selection)
		print('Max: ', max_val)
		print('Min: ', min_val)
		print('Mean: ', mean_val)
		print('Standard Deviation: ', stdev)
		print()

		# 5NN with Feature Selection
		five_nn_accuracies_with_feature_selection = \
			ten_fold_cross_validation(data_with_feature_selection, knn, k=5)
		max_val, min_val, mean_val, stdev = calculate_statistics(five_nn_accuracies_with_feature_selection)
		print("5NN with Feature Selection")
		print('Accuracies: ', five_nn_accuracies_with_feature_selection)
		print('Max: ', max_val)
		print('Min: ', min_val)
		print('Mean: ', mean_val)
		print('Standard Deviation: ', stdev)
		print()

		# Naive Bayes without Feature Selection
		nb_accuracies_without_feature_selection = \
			ten_fold_cross_validation(data_without_feature_selection, nb)
		max_val, min_val, mean_val, stdev = calculate_statistics(nb_accuracies_without_feature_selection)
		print("Naive Bayes without Feature Selection")
		print('Accuracies: ', nb_accuracies_without_feature_selection)
		print('Max: ', max_val)
		print('Min: ', min_val)
		print('Mean: ', mean_val)
		print('Standard Deviation: ', stdev)
		print()

		# Naive Bayes with Feature Selection
		nb_accuracies_with_feature_selection = \
			ten_fold_cross_validation(data_with_feature_selection, nb)
		max_val, min_val, mean_val, stdev = calculate_statistics(nb_accuracies_with_feature_selection)
		print("Naive Bayes with Feature Selection")
		print('Accuracies: ', nb_accuracies_with_feature_selection)
		print('Max: ', max_val)
		print('Min: ', min_val)
		print('Mean: ', mean_val)
		print('Standard Deviation: ', stdev)
		print()

		

