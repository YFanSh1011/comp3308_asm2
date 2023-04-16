import math

def mean(data):
    curr_sum = 0
    for num in data:
        curr_sum += num
    return curr_sum / len(data)

def stdev(data, data_mean):
    sum_of_squares = 0
    for num in data:
        sum_of_squares += (num - data_mean) ** 2
    return math.sqrt(sum_of_squares / (len(data) - 1))

def density_function(data, mean_val, stdev_val):
    first_comp = 1 / ((stdev_val) * math.sqrt(2 * math.pi))
    second_comp = math.exp(-(((data - mean_val) ** 2) / (2 * stdev_val * stdev_val)))
    return first_comp * second_comp

def classify_nb_f(training_filename, testing_filename):
    # Read the dataset and convert them into correct data format:
    training_data = []
    testing_data = []
    with open(training_filename, 'r') as f:
        training_data_string = [line for line in f.read().split() if len(line) != 0]
        for line in training_data_string:
            segs = line.strip().split(',')
            # Converting all entries to float and keep the last one being string
            data_entry = [float(segs[i]) for i in range(len(segs) - 1)]
            data_entry.append(segs[-1])
            training_data.append(data_entry)
            
    with open(testing_filename, 'r') as f:
        testing_data_string = [line for line in f.read().split() if len(line) != 0]
        for line in testing_data_string:
            segs = line.strip().split(',')
            data_entry = [float(segs[i]) for i in range(len(segs))]
            testing_data.append(data_entry)
    
    # Calculate the mean and standard deviation for each metrics:
    yes_mean_values = []
    yes_stdev_values = []
    yes_count = 0
    no_mean_values = []
    no_stdev_values = []
    no_count = 0
    
    for i in range(len(training_data[0]) - 1):
        # Initialise empty array to store values for current metric
        yes_curr_metric = []
        no_curr_metric = []
        
        for entry in training_data:
            # Kinda a mistake for counting "yes" and "no" multiple times
            # But since the ratio is that same, I'll leave it like this
            # i.e., count_yes / (count_yes + count_no) == count_yes*n / (count_yes*n + count_no*n)
            if entry[-1].strip() == 'yes':
                yes_curr_metric.append(entry[i])
                yes_count += 1
            else:
                no_curr_metric.append(entry[i])
                no_count += 1

        # Append the calculated values
        yes_mean = mean(yes_curr_metric)
        yes_mean_values.append(yes_mean)
        yes_stdev_values.append(stdev(yes_curr_metric, yes_mean))
        
        no_mean = mean(no_curr_metric)
        no_mean_values.append(no_mean)
        no_stdev_values.append(stdev(no_curr_metric, no_mean))
    
    predictions = []

    # Calculate the corresponding probablilities based on the density function
    for testing_data_entry in testing_data:
        yes_prob = 1
        no_prob = 1
        for i in range(len(testing_data_entry)):
            yes_prob *= density_function(testing_data_entry[i], yes_mean_values[i], yes_stdev_values[i])
            no_prob *= density_function(testing_data_entry[i], no_mean_values[i], no_stdev_values[i])
        yes_prob *= (yes_count)/(yes_count + no_count)
        no_prob *= (no_count)/(yes_count + no_count)
        predictions.append("yes" if yes_prob >= no_prob else "no")
        
    return predictions

def classify_nb(training_data, testing_data):
    # Calculate the mean and standard deviation for each metrics:
    yes_mean_values = []
    yes_stdev_values = []
    yes_count = 0
    no_mean_values = []
    no_stdev_values = []
    no_count = 0
    
    for entry in training_data:
        if entry[-1].strip() == 'yes':
            yes_count += 1
        else:
            no_count += 1

    for i in range(len(training_data[0]) - 1):
        # Initialise empty array to store values for current metric
        yes_curr_metric = []
        no_curr_metric = []

        for entry in training_data:
            # Kinda a mistake for counting "yes" and "no" multiple times
            # But since the ratio is that same, I'll leave it like this
            # i.e., count_yes / (count_yes + count_no) == count_yes*n / (count_yes*n + count_no*n)
            if entry[-1].strip() == 'yes':
                yes_curr_metric.append(entry[i])
            else:
                no_curr_metric.append(entry[i])

        # Append the calculated values
        yes_mean = mean(yes_curr_metric)
        yes_mean_values.append(yes_mean)
        yes_stdev_values.append(stdev(yes_curr_metric, yes_mean))
        
        no_mean = mean(no_curr_metric)
        no_mean_values.append(no_mean)
        no_stdev_values.append(stdev(no_curr_metric, no_mean))
    
    accuracte_prediction_count = 0

    # Calculate the corresponding probablilities based on the density function
    for testing_data_entry in testing_data:
        yes_prob = 1
        no_prob = 1
        for i in range(len(testing_data_entry) - 1):
            yes_prob *= density_function(testing_data_entry[i], yes_mean_values[i], yes_stdev_values[i])
            no_prob *= density_function(testing_data_entry[i], no_mean_values[i], no_stdev_values[i])
        yes_prob *= (yes_count)/(yes_count + no_count)
        no_prob *= (no_count)/(yes_count + no_count)
        
        if (yes_prob >= no_prob and testing_data_entry[-1] == "yes") \
            or (yes_prob < no_prob and testing_data_entry[-1] == "no"):
            accuracte_prediction_count += 1
        
    return accuracte_prediction_count / len(testing_data)


if __name__ == "__main__":
    print(classify_nb_f("training.csv", "testing.csv"))