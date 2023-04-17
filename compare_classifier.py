import math
import json
import sys
from scipy.stats import t


def calculate_statistical_significance(data_set1, data_set2, confidence_level=0.95):
	# Calculate the differences between the accuracy of each folds:
	differences = []
	for i in range(len(data_set1)):
		differences.append(abs(data_set1[i] - data_set2[i]))
	mean_diff = sum(differences) / len(differences)

	# Calculate the standard deviation of the differences:
	sum_of_squares = 0
	for num in differences:
		sum_of_squares += (num - mean_diff) ** 2
	stdev = math.sqrt(sum_of_squares / (len(differences) - 1))

	# Calculate the z-score:
	z_score_upper = mean_diff + abs(t.ppf((1 + confidence_level) / 2, len(data_set1) - 1) \
		* (stdev / math.sqrt(len(differences))))
	z_score_lower = mean_diff - abs(t.ppf((1 + confidence_level) / 2, len(data_set1) - 1) \
		* (stdev / math.sqrt(len(differences))))
	
	print("Z-score: ({}, {})".format(z_score_lower, z_score_upper))
	return (z_score_upper >= 0 and  0 >= z_score_lower)


if __name__ == "__main__":
	# Load data from json file
	try:
		data_dict = json.load(open('Results/calculation_results.json', 'r'))
	except FileNotFoundError:
		print("File not found. Please run calculate_results.py first.")
		exit()
	
	one_nn_wo_fs = data_dict['1NN without Feature Selection']["Accuracies"]
	one_nn_w_fs = data_dict['1NN with Feature Selection']["Accuracies"]

	five_nn_wo_fs = data_dict['5NN without Feature Selection']["Accuracies"]
	five_nn_w_fs = data_dict['5NN with Feature Selection']["Accuracies"]
	
	nb_wo_fs = data_dict['Naive Bayes without Feature Selection']["Accuracies"]
	nb_w_fs = data_dict['Naive Bayes with Feature Selection']["Accuracies"]
	
	with open("Results/comparison_result.txt", "w") as sys.stdout:
		print("------------------------")
		print("Vertical Comparisons")
		print("------------------------")
		# Compare the result with or without feature selection for 1NN
		print('1NN without Feature Selection vs 1NN with Feature Selection:')
		print("There {} a statistical significance between the two algorithms.".format("IS" if calculate_statistical_significance(one_nn_wo_fs, one_nn_w_fs) else "IS NOT"))
		print()

		# Compare the result with or without feature selection for 5NN
		print('5NN without Feature Selection vs 5NN with Feature Selection:')
		print("There {} a statistical significance between the two algorithms.".format("IS" if calculate_statistical_significance(five_nn_wo_fs, five_nn_w_fs) else "IS NOT"))
		print()

		# Compare the result with or without feature selection for Naive Bayes
		print('Naive Bayes without Feature Selection vs Naive Bayes with Feature Selection:')
		print("There {} a statistical significance between the two algorithms.".format("IS" if calculate_statistical_significance(nb_wo_fs, nb_w_fs) else "IS NOT"))
		print()
		print()

		print("------------------------")
		print("Horizontal Comparisons Without Feature Selection")
		print("------------------------")
		# Compare the result of 1NN with 5NN
		print('1NN without Feature Selection vs 5NN without Feature Selection:')
		print("There {} a statistical significance between the two algorithms.".format("IS" if calculate_statistical_significance(one_nn_wo_fs, five_nn_wo_fs) else "IS NOT"))
		print()

		# Compare the result of 1NN with Naive Bayes
		print('1NN without Feature Selection vs Naive Bayes without Feature Selection:')
		print("There {} a statistical significance between the two algorithms.".format("IS" if calculate_statistical_significance(one_nn_wo_fs, nb_wo_fs) else "IS NOT"))
		print()

		# Compare the result of 5NN with Naive Bayes
		print('5NN without Feature Selection vs Naive Bayes without Feature Selection:')
		print("There {} a statistical significance between the two algorithms.".format("IS" if calculate_statistical_significance(five_nn_wo_fs, nb_wo_fs) else "IS NOT"))
		print()
		print()
		
		print("------------------------")
		print("Horizontal Comparison With Feature Selection")
		print("------------------------")
		# Compare the result of 1NN with 5NN
		print('1NN with Feature Selection vs 5NN with Feature Selection:')
		print("There {} a statistical significance between the two algorithms.".format("IS" if calculate_statistical_significance(one_nn_w_fs, five_nn_w_fs) else "IS NOT"))
		print()

		# Compare the result of 1NN with Naive Bayes
		print('1NN with Feature Selection vs Naive Bayes with Feature Selection:')
		print("There {} a statistical significance between the two algorithms.".format("IS" if calculate_statistical_significance(one_nn_w_fs, nb_w_fs) else "IS NOT"))
		print()

		# Compare the result of 5NN with Naive Bayes
		print('5NN with Feature Selection vs Naive Bayes with Feature Selection:')
		print("There {} a statistical significance between the two algorithms.".format("IS" if calculate_statistical_significance(five_nn_w_fs, nb_w_fs) else "IS NOT"))
		print()