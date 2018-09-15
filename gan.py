from random import random
from time import time
from os import mkdir
from os.path import isfile
import csv
import tensorflow as tf

BATCH_SIZE = 32
NUM_ITERATIONS = 2001
PRINT_FREQUENCY = NUM_ITERATIONS // 5
NUM_FEATURES = 8
DATASET_FILE = 'CSC1015.csv'
FOLDER = 'CSC1015'
QUIET_TRAINING = True
ROWS_TO_GENERATE = 1000

# Function to load and convert DATASET_FILE into a 2d array of results between 0 and 1
def load_dataset():
    file = open(DATASET_FILE)
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

# Helper function to pretty print generator output
def pretty_print_output(current):
    print('-------------------------')
    print('FAScore:      {:.0f}'.format(current[0] * 600))
    print('Gr12 English: {:.0f}'.format(current[1] * 100))
    print('Gr12 Math:    {:.0f}'.format(current[2] * 100))
    print('Gr12 Physics: {:.0f}'.format(current[3] * 100))
    print('NBT AL:       {:.0f}'.format(current[4] * 100))
    print('NBT QL:       {:.0f}'.format(current[5] * 100))
    print('NBT Math:     {:.0f}'.format(current[6] * 100))
    print()
    print('Result:       {:.0f}'.format(current[7] * 100))
    print('-------------------------')

# Internal function which takes a session, latent input placeholder, latent vector size and 
# filename and will generate 1000 rows of data. Requires a pretrained GAN. Saves data in 
# 'Generated_Data/' + FOLDER + '/' filename + '.csv'
def generate(sess, Y, G_sample, latent_vector_size, filename):    
    should_print = False # input('Print Data? (Y/N): ').upper()[0]  == 'Y'
    filename = 'Generated_Data/{}/{}.csv'.format(FOLDER, filename)
    quantity = ROWS_TO_GENERATE # int(input('Number of rows to generate: '))
    
    latent_vector = [[random() for x in range(latent_vector_size)] for i in range(quantity)]
    samples_fake = sess.run(G_sample, feed_dict={Y: latent_vector})
    
    rows = []
    for i in range(quantity):
        current = samples_fake[i]
        output = []
        output.append('GENERATED')
        output.append('{:.0f}'.format(current[0] * 600))
        output.append('{:.0f}'.format(current[1] * 100))
        output.append('{:.0f}'.format(current[2] * 100))
        output.append('{:.0f}'.format(current[3] * 100))
        output.append('{:.0f}'.format(current[4] * 100))
        output.append('{:.0f}'.format(current[5] * 100))
        output.append('{:.0f}'.format(current[6] * 100))
        output.append('{}'.format(FOLDER))
        output.append('{:.0f}'.format(current[7] * 100))
        rows.append(output)
        
        if not should_print:
            continue
        pretty_print_output(samples_fake[0])

    
    min_v = [min([sample[i] for sample in samples_fake]) for i in range(NUM_FEATURES)]
    max_v = [max([sample[i] for sample in samples_fake]) for i in range(NUM_FEATURES)]
    range_v = [max_v[i]-min_v[i] for i in range(NUM_FEATURES)]
    ave_v = [sum([sample[i] for sample in samples_fake]) / len(samples_fake) for i in range(NUM_FEATURES)]
        
    rows.sort(key=lambda x: x[9])

    file = open(filename, 'w', newline='')
    file_writer = csv.writer(file)
    file_writer.writerow(['id','faScore','english','maths','physics','nbtAL','nbtQL','nbtMath','course','result'])
    file_writer.writerows(rows)
    file.close()
    
    return range_v, ave_v

# Internal function to build a discriminator of len(layer_sizes) hidden layers with layer_sizes[i] 
# hidden nodes in each layer
def create_discriminator_model(layer_sizes):
    D_model = tf.keras.Sequential()
    for index, layer_size in enumerate(layer_sizes):
        layer = tf.keras.layers.LeakyReLU(layer_size)
        if index == 0:
            layer = tf.keras.layers.LeakyReLU(layer_size, input_shape=(NUM_FEATURES,))
        D_model.add(layer)
    D_model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    return D_model

# Internal function to build a generator of len(layer_sizes) hidden layers with layer_sizes[i] 
# hidden nodes in each layer with latent_vector_size input nodes
def create_generator_model(layer_sizes, latent_vector_size):
    G_model = tf.keras.Sequential()
    for index, layer_size in enumerate(layer_sizes):
        layer = tf.keras.layers.Dense(layer_size,
                                      activation="relu")
        if index == 0:
            layer = tf.keras.layers.Dense(layer_size, 
                                          activation="relu",
                                          input_shape=(latent_vector_size,))
        G_model.add(layer)
    G_model.add(tf.keras.layers.Dense(NUM_FEATURES, activation="sigmoid"))
    return G_model

# Helper method to convert a file name to network configuration
def parse(filename):
    vars = filename.split('-')
    if len(vars) < 4:
        return 0
    d = [int(x) for x in vars[0][1:].split('_')]
    g = [int(x) for x in vars[1][1:].split('_')]
    l = int(vars[2][1:])
    r = int(vars[3])
    return d, g, l, r

# Main method to build and train a GAN network. Will train with dataset as training data and 
# will save network to './Models/' + FOLDER + '/' + filename. Will generate data using generate
def train_gan(d_layer_sizes, g_layer_sizes, dataset, latent_vector_size, filename):
    print('Training with settings: {}'.format(filename))
    start_time = time() 
    tf.reset_default_graph()
    train_data_iterator = dataset.make_one_shot_iterator()
    
    D_model = create_discriminator_model(d_layer_sizes)        
    G_model =create_generator_model(g_layer_sizes, latent_vector_size)
    
    X = tf.placeholder(tf.float32, shape=[None, NUM_FEATURES])
    true_data = train_data_iterator.get_next()
    Y = tf.placeholder(tf.float32, shape=[None, latent_vector_size], name='generator_input')

    G_model_out = G_model(Y)
    G_sample = tf.identity(G_model_out, name="G_sample")
    D_test = D_model(X)
    D_real = D_model(true_data)
    D_fake = D_model(G_sample)
    D_logits_real = -tf.log(1/D_real - 1)
    D_logits_fake = -tf.log(1/D_fake - 1)
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_real, labels=tf.ones_like(D_logits_real)))
    D_loss_generated = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_fake, labels=tf.zeros_like(D_logits_fake)))
    D_loss = D_loss_real + D_loss_generated
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=-D_logits_fake, labels=tf.zeros_like(D_logits_fake)))

    D_solver = tf.train.GradientDescentOptimizer(0.01).minimize(D_loss, var_list=D_model.variables)
    G_solver = tf.train.GradientDescentOptimizer(0.01).minimize(G_loss, var_list=G_model.variables)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    for it in range(NUM_ITERATIONS):
        latent_vector = [[random() for x in range(latent_vector_size)] for i in range(BATCH_SIZE)]
        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={Y: latent_vector})
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Y: latent_vector})

        if not QUIET_TRAINING and it % PRINT_FREQUENCY == 0:
            latent_vector = [[random() for x in range(latent_vector_size)]]
            samples_fake = sess.run(G_sample, feed_dict={Y: latent_vector})
            D_sample_real = sess.run(D_real)
            D_sample_fake = sess.run(D_test, feed_dict={X: samples_fake})

            print('-------------------------')
            print('Epoch: {}'.format(it))
            print('D loss: {:.4}'.format(D_loss_curr))
            print('G_loss: {:.4}'.format(G_loss_curr))
            print('D_sample_real: {}'.format(D_sample_real[0][0]))
            print('D_sample_fake: {}'.format(D_sample_fake[0][0]))
            print()
            print('Generated Sample')
            pretty_print_output(samples_fake[0])
            print('-------------------------')
    
    model_dir = './Models/' + FOLDER + '/' + filename
    try:
        mkdir(model_dir)
    except FileExistsError:
        pass
    model_file = model_dir + '/model.ckpt'
    saver = tf.train.Saver()
    saver.save(sess, model_file)
    
    range_v, ave_v = generate(sess, Y, G_sample, latent_vector_size, filename)
    f = open(model_dir+'/stats.txt', 'w')
    f.write('Averages: {}\n'.format(ave_v))
    f.write('Ranges: {}'.format(range_v))
    f.close()
    
    sess.close()
    print('Completed training in {:.2f}s'.format(time() - start_time))
    print('Averages: {}'.format([round(x * 100) for x in ave_v]))
    print('Ranges:   {}'.format([round(x * 100) for x in range_v]))

# Method to load a GAN saved in 'filename ' folder
def load_gan(filename):
    model_dir = './Models/' + FOLDER + '/' + filename + '/model.ckpt'
    _, layer_sizes, latent_vector_size, _ = parse(filename)
    
    sess = tf.Session()    
    saver = tf.train.import_meta_graph(model_dir + '.meta')
    saver.restore(sess, model_dir)
    graph = tf.get_default_graph()
    Y = graph.get_tensor_by_name("generator_input:0")
    G_sample = graph.get_tensor_by_name("G_sample:0")
    
    generate(sess, Y, G_sample, latent_vector_size, filename+'a')

def main():
    layer_sizes = [2**x for x in range(3, 7)]
    depth_1_layers = [[x] for x in layer_sizes]
    depth_2_layers = [[x, y] for x in layer_sizes for y in layer_sizes]
    depth_3_layers = [[x, y, z] for x in layer_sizes for y in layer_sizes for z in layer_sizes]
    layer_layouts = depth_1_layers + depth_2_layers # + depth_3_layers[0:len(depth_3_layers)//4]
    latent_vector_sizes = [2**x for x in range(5, 8)]
    repitions = 3
    
    start_time = time()
    
    raw_data = load_dataset()
    data = tf.convert_to_tensor(raw_data, dtype=tf.float32)
    train_dataset = tf.data.Dataset.from_tensor_slices(data)
    train_dataset = train_dataset.shuffle(buffer_size=1000)
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.repeat()
    
    min_v = [min([row[i] for row in raw_data]) for i in range(NUM_FEATURES)]
    max_v = [max([row[i] for row in raw_data]) for i in range(NUM_FEATURES)]
    range_v = [max_v[i]-min_v[i] for i in range(NUM_FEATURES)]
    ave_v = [sum([row[i] for row in raw_data]) / len(raw_data) for i in range(NUM_FEATURES)]
    
    print('Original Data Averages: {}'.format([round(x * 100) for x in ave_v]))
    print('Original Data Ranges:   {}'.format([round(x * 100) for x in range_v]))
    print('Number of iterations to do: {}'.format(len(layer_layouts) ** 2 * len(latent_vector_sizes) * repitions))
    
    for index, d_layer in enumerate(layer_layouts):
        print('\n  {}% complete'.format(float(index) / len(layer_layouts) * 100))
        print('  Original Data Averages: {}'.format([round(x * 100) for x in ave_v]))
        print('  Original Data Ranges:   {}'.format([round(x * 100) for x in range_v]))
        count = 0
        for g_layer in layer_layouts:
            print()
            for latent_vector_size in latent_vector_sizes:
                for repition in range(repitions):
                    d_layout = 'D' + '_'.join([str(x) for x in d_layer])
                    g_layout = 'G' + '_'.join([str(x) for x in g_layer])
                    filename = '{}-{}-L{}-{}'.format(d_layout, g_layout, latent_vector_size, repition)
                    if isfile('Generated_Data/{}/{}.csv'.format(FOLDER, filename)):
                        continue
                    train_gan(d_layer, g_layer, train_dataset, latent_vector_size, filename)
    print(count)                
    
    time_elapsed = time() - start_time
    h = int(time_elapsed / (60 * 60))
    m = int((time_elapsed % (60 * 60)) / 60)
    s = time_elapsed % 60
    print('Completed all training in {}:{:>02}:{:>05.2f}'.format(h, m, s))

if __name__ == '__main__':
    main()
    # load_gan('D16-G16-L64-0')
    # Invert the commented status of the above lines to generate data without retraining