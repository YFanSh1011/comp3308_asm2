import math

def calculate_distance(source, target):
    sum_of_squares = 0
    for i in range(len(target)):
        if (type(target[i]) != str):
            sum_of_squares += (source[i] - target[i]) ** 2
    return math.sqrt(sum_of_squares)
    

def classify_nn_f(training_filename, testing_filename, k):
    # Read the dataset and convert them into correct data format:
    training_data = []
    testing_data = []
    with open(training_filename, 'r') as f:
        training_data_string = f.readlines()
        for line in training_data_string:
            segs = line.strip().split(',')
            # Converting all entries to float and keep the last one being string
            data_entry = [float(segs[i]) for i in range(len(segs) - 1) if i != len(segs) - 1]
            data_entry.append(segs[-1])
            training_data.append(data_entry)
    with open(testing_filename, 'r') as f:
        testing_data_string = f.readlines()
        for line in testing_data_string:
            segs = line.split(',')
            data_entry = [float(segs[i]) for i in range(len(segs))]
            testing_data.append(data_entry)
    
    # For each training data point, initialise an array, store the calculate the distance
    # the sort the array and pick the k closest data points
    # Format: (distance, classifier)
    predictions = []
    for data_point in testing_data:
        distances = []
        for training_point in training_data:
            distances.append((calculate_distance(training_point, data_point), training_point[-1]))
        distances.sort(key=lambda x: x[0])
        selected_points = distances[:k]
        count_yes = 0
        for point in selected_points:
            if point[1] == 'yes':
                count_yes += 1
            else:
                count_yes -= 1
        predictions.append("yes" if count_yes >= 0 else "no")
    
    return predictions


def classify_nn(training_data, testing_data, k):
    # For each training data point, initialise an array, store the calculate the distance
    # the sort the array and pick the k closest data points
    # Format: (distance, classifier)
    accurate_prediction_count = 0
    confusion_matrix = {"True Positive": 0, "False Positive": 0, "True Negative": 0, "False Negative": 0}

    for data_point in testing_data:
        distances = []
        for training_point in training_data:
            distances.append((calculate_distance(training_point, data_point), training_point[-1]))
        distances.sort(key=lambda x: x[0])
        selected_points = distances[:k]
        count_yes = 0
        for point in selected_points:
            if point[1] == 'yes':
                count_yes += 1
            else:
                count_yes -= 1
        if count_yes >= 0 and data_point[-1] == 'yes':
            accurate_prediction_count += 1
            confusion_matrix["True Positive"] += 1
        elif count_yes >= 0 and data_point[-1] == 'no':
            confusion_matrix["False Positive"] += 1
        elif count_yes < 0 and data_point[-1] == 'yes':
            confusion_matrix["False Negative"] += 1
        else:
            accurate_prediction_count += 1
            confusion_matrix["True Negative"] += 1
            
    return accurate_prediction_count / len(testing_data), confusion_matrix

if __name__ == "__main__":
    print(classify_nn_f("training.csv", "testing.csv", 3))
    