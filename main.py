from Algorithms.knn import classify_nn as knn
from Algorithms.nb import classify_nb as nb


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


def ten_fold_cross_validation(data, algorithm, knn_k):
	accuracies = []
	for i in range(len(data)):
		print("Current fold: ", i+1)
		train, test = train_test_split(data, i)
		accuracies.append(algorithm(train, test, knn_k))
	return accuracies

if __name__ == '__main__':
	data = split_data('Data/pima-10-folds.csv')

	# 1NN
	one_nn_accuracies = ten_fold_cross_validation(data, knn, 1)
	print('1-NN: ', one_nn_accuracies)

	# 5NN
	five_nn_accuracies = ten_fold_cross_validation(data, knn, 5)
	print('5-NN: ', five_nn_accuracies)
	


	