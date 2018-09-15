from os import listdir
from math import sqrt
from shutil import copyfile
import csv

DATASET = 'CSC1015'
TOP_N = 50

# Function to load and convert 'DATASET.csv' into a 2d array of results between 0 and 1
def load_dataset():
    file = open('{}.csv'.format(DATASET))
    data_reader = csv.reader(file, delimiter=';')
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

# Distance funtion to find Euclidan Distance between 2 arrays
def dist(arr1, arr2):
    return sqrt(sum([(arr1[i] - arr2[i])**2 for i in range(8)]))

# Function to loop through all Models stored in Model/DATASET and compute the distance of their stats to the original stats
def calculate_all_distances(original_range, original_average):
    results = []    
    for model in listdir('Models/{}'.format(DATASET)):
        stats = open('Models/{}/{}/stats.txt'.format(DATASET, model), 'r')
        line1 = stats.readline()
        if len(line1) == 0:
            continue
        line2 = stats.readline()
        average = eval(line1.split(':')[1][1:])
        range = eval(line2.split(':')[1][1:])
        range_dist =  dist(range, original_range)
        ave_dist = dist(average, original_average)
        results.append({'name': model, 'range': range, 'ave':average, 'range_dist':range_dist, 'ave_dist':ave_dist})
        stats.close()
        
    return results
    
# Helper method to pretty print top n results with a heading
def print_top_n(n, results, heading):
    print('')
    print(heading)
    print('{: <22}: {}   {}'.format('Data', 'Avg', 'Rng'))
    for i in range(n):
        result = results[i]
        print('{: <22}: {:.2f} {:.2f}'.format(result['name'], result['ave_dist'] * 100, result['range_dist'] * 100))
        
def main():
    raw_data = load_dataset()
    min_v = [min([row[i] for row in raw_data]) for i in range(8)]
    max_v = [max([row[i] for row in raw_data]) for i in range(8)]
    range_v = [max_v[i]-min_v[i] for i in range(8)]
    ave_v = [sum([row[i] for row in raw_data]) / len(raw_data) for i in range(8)]
    
    results = calculate_all_distances(range_v, ave_v)
    
    print('Number of models analyzed: {}'.format(len(results)))
    shared = []
    min_range = []
    results.sort(key=lambda x: x['range_dist'])
    print_top_n(TOP_N//5, results, 'Sorted by distance to true range')
    
    for i in range(TOP_N):
        min_range.append(results[i]['name'])
        
    results.sort(key=lambda x: x['ave_dist'])
    print_top_n(TOP_N//5, results, 'Sorted by distance to true average')
        
    for i in range(TOP_N):
        if results[i]['name'] in min_range:
            shared.append(i)
    
    print('\nData in top {} of both'.format(TOP_N))
    print('{: <22}: {}   {}'.format('Data', 'Avg', 'Rng'))
    for idx, i in enumerate(shared):
        name = results[i]['name']
        print('{: <22}: {:.2f} {:.2f}'.format(name, results[i]['ave_dist'] * 100, results[i]['range_dist'] * 100))
        copyfile('Generated_Data/{}/{}.csv'.format(DATASET, name), '{}-{}.csv'.format(idx, name))
        
if __name__=='__main__':
    main()