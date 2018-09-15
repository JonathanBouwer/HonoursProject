import tensorflow as tf
from math import floor
from os import listdir
import csv

DATASET = 'CSC1016'
INPUT_VECTOR_SIZE = 7
OUTPUT_VECTOR_SIZE = 2
NETWORK_LAYER_1_SIZE = 64
NETWORK_LAYER_2_SIZE = 64

# Function to load and convert DATASET into a 2d array of results between 0 and 1
def load_dataset(filename, delim=';'):
    file = open(filename)
    data_reader = csv.reader(file, delimiter=delim)
    data = []
    header = next(data_reader)
    for data_line in data_reader:
        #id     = data_line[0]
        faScore = float(data_line[1]) / 600.0
        english = float(data_line[2]) / 100.0
        maths   = float(data_line[3]) / 100.0
        physics = float(data_line[4]) / 100.0
        nbtAL   = float(data_line[5]) / 100.0
        nbtQL   = float(data_line[6]) / 100.0
        nbtMath = float(data_line[7]) / 100.0
        #course = data_line[8]
        result  = float(data_line[9]) / 100.0
        parsed_data = [faScore, english, maths, physics, nbtAL, nbtQL, nbtMath, result]
        if min(parsed_data) < 0.1 or max(parsed_data) > 1: # Filter data_lines
            continue
        data.append(parsed_data)
    file.close()
    return data

# Helper function to pretty print accuracy of trained network
def pretty_print_accuracy(correct_pass, pass_count, correct_fail, fail_count):
    print('Passes: {:>4} / {:>4} = {:.2f}%'.format(correct_pass, pass_count, correct_pass / pass_count * 100))
    print('Fails:  {:>4} / {:>4} = {:.2f}%'.format(correct_fail, fail_count, correct_fail / fail_count * 100))
    num = correct_pass + correct_fail
    denom = pass_count + fail_count
    print('All:    {:>4} / {:<4} = {:.2f}%'.format(num, denom, num / denom * 100))

# Method to evaluate a model against a given dataset
def evaluate_model(model, dataset):
    correct_pass = 0
    pass_count = 0
    correct_fail = 0
    fail_count = 0
    row = tf.convert_to_tensor([x[:-1] for x in dataset], tf.float32)
    predictions = model.predict(row, steps=1)
    for index, prediction in enumerate(predictions):
        passed = 1 if dataset[index][-1] >= 0.5 else 0
        predicted_passed = 1 if prediction[0] < prediction[1] else 0
        correct = (passed and predicted_passed) or (not passed and not predicted_passed)
        if passed:
            pass_count += 1
            if correct:
                correct_pass +=1
        else:
            fail_count += 1
            if correct:
                correct_fail +=1
    pretty_print_accuracy(correct_pass, pass_count, correct_fail, fail_count)

# Method to train a model with given training data
def train_model(training_data):
    model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(NETWORK_LAYER_1_SIZE, activation=tf.nn.relu, input_shape=(INPUT_VECTOR_SIZE,)),
      tf.keras.layers.Dense(NETWORK_LAYER_2_SIZE, activation=tf.nn.relu),
      tf.keras.layers.Dense(OUTPUT_VECTOR_SIZE, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    original_x = tf.convert_to_tensor([x[:-1] for x in training_data], dtype=tf.float32)
    original_temp_labels = [[min(int(floor(x[-1]*OUTPUT_VECTOR_SIZE)), OUTPUT_VECTOR_SIZE-1)] for x in training_data]
    original_labels = tf.convert_to_tensor(original_temp_labels , dtype=tf.float32)
    
    model.fit(original_x, original_labels, epochs=8, steps_per_epoch=512, verbose=0)
    return model

# Helper method to call train_model and evaluate_model
def train_and_evaluate_model(train_dataset, evaluate_dataset, dataset_name):
    print(dataset_name)
    model = train_model(train_dataset)
    evaluate_model(model, evaluate_dataset)
 
def main():
    original_data = load_dataset('{}.csv'.format(DATASET))
    train_and_evaluate_model(original_data, original_data, 'Training data')
    
    filenames = listdir('Test_Data/{}'.format(DATASET))
    for filename in filenames:
        dataset = load_dataset('Test_Data/{}/{}'.format(DATASET, filename), ',')
        train_and_evaluate_model(dataset, original_data, filename[filename.find('-')+1:-6])
    
if __name__=='__main__':
    main()