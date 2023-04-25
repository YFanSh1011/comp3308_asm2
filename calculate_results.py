import math
import json
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
	confusion_matrices = []
	for i in range(len(data)):
		train, test = train_test_split(data, i)
		if algorithm == nb:
			accuracy, cm = algorithm(train, test)
			accuracies.append(accuracy)
			confusion_matrices.append(cm)
		else:
			accuracy, cm = algorithm(train, test, kwargs['k'])
			accuracies.append(accuracy)
			confusion_matrices.append(cm)
	
	# Computer average of confusion matrices
	confusion_matrix = {"True Positive": 0, "False Positive": 0, "True Negative": 0, "False Negative": 0}
	for cm in confusion_matrices:
		for key in confusion_matrix:
			confusion_matrix[key] += cm[key]

	return accuracies, confusion_matrix


if __name__ == '__main__':
	data_without_feature_selection = split_data('Data/pima-10-folds.csv')
	data_with_feature_selection = split_data('Data/pima-CFS-folds.csv')
	result_dict = {}

	
	# 1NN without Feature Selection
	one_nn_accuracies_without_feature_selection, one_nn_wo_fs_cm = \
		ten_fold_cross_validation(data_without_feature_selection, knn, k=1)
	max_val, min_val, mean_val, stdev = calculate_statistics(one_nn_accuracies_without_feature_selection)
	result_dict["1NN without Feature Selection"] = {
		"Accuracies": one_nn_accuracies_without_feature_selection,
		"Max": max_val,
		"Min": min_val,
		"Mean": mean_val,
		"Standard Deviation": stdev,
		"Confusion Matrix": one_nn_wo_fs_cm
	}
	print("1NN without Feature Selection Accuracies Calculated")

	# 1NN with Feature Selection
	one_nn_accuracies_with_feature_selection, one_nn_w_fs_cm = \
		ten_fold_cross_validation(data_with_feature_selection, knn, k=1)
	max_val, min_val, mean_val, stdev = calculate_statistics(one_nn_accuracies_with_feature_selection)
	result_dict["1NN with Feature Selection"] = {
		"Accuracies": one_nn_accuracies_with_feature_selection,
		"Max": max_val,
		"Min": min_val,
		"Mean": mean_val,
		"Standard Deviation": stdev,
		"Confusion Matrix": one_nn_w_fs_cm
	}
	print("1NN with Feature Selection Accuracies Calculated")

	# 5NN without Feature Selection
	five_nn_accuracies_without_feature_selection, five_nn_wo_fs_cm = \
		ten_fold_cross_validation(data_without_feature_selection, knn, k=5)
	max_val, min_val, mean_val, stdev = calculate_statistics(five_nn_accuracies_without_feature_selection)
	result_dict["5NN without Feature Selection"] = {
		"Accuracies": five_nn_accuracies_without_feature_selection,
		"Max": max_val,
		"Min": min_val,
		"Mean": mean_val,
		"Standard Deviation": stdev,
		"Confusion Matrix": five_nn_wo_fs_cm
	}
	print("5NN without Feature Selection Accuracies Calculated")

	# 5NN with Feature Selection
	five_nn_accuracies_with_feature_selection, five_nn_w_fs_cm = \
		ten_fold_cross_validation(data_with_feature_selection, knn, k=5)
	max_val, min_val, mean_val, stdev = calculate_statistics(five_nn_accuracies_with_feature_selection)
	result_dict["5NN with Feature Selection"] = {
		"Accuracies": five_nn_accuracies_with_feature_selection,
		"Max": max_val,
		"Min": min_val,
		"Mean": mean_val,
		"Standard Deviation": stdev,
		"Confusion Matrix": five_nn_w_fs_cm
	}	
	print("5NN with Feature Selection Accuracies Calculated")

	# Naive Bayes without Feature Selection
	nb_accuracies_without_feature_selection, nb_wo_fs_cm = \
		ten_fold_cross_validation(data_without_feature_selection, nb)
	max_val, min_val, mean_val, stdev = calculate_statistics(nb_accuracies_without_feature_selection)
	result_dict["Naive Bayes without Feature Selection"] = {
		"Accuracies": nb_accuracies_without_feature_selection,
		"Max": max_val,
		"Min": min_val,
		"Mean": mean_val,
		"Standard Deviation": stdev,
		"Confusion Matrix": nb_wo_fs_cm
	}
	print("Naive Bayes without Feature Selection Accuracies Calculated")

	# Naive Bayes with Feature Selection
	nb_accuracies_with_feature_selection, nb_w_fs_cm = \
		ten_fold_cross_validation(data_with_feature_selection, nb)
	max_val, min_val, mean_val, stdev = calculate_statistics(nb_accuracies_with_feature_selection)
	result_dict["Naive Bayes with Feature Selection"] = {
		"Accuracies": nb_accuracies_with_feature_selection,
		"Max": max_val,
		"Min": min_val,
		"Mean": mean_val,
		"Standard Deviation": stdev,
		"Confusion Matrix": nb_w_fs_cm
	}
	print("Naive Bayes with Feature Selection Accuracies Calculated")

	# Write results to file
	with open('Results/calculation_results.json', 'w') as f:
		json.dump(result_dict, f, indent=4)
		print("Results written to file")