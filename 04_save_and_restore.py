import tensorflow as tf 
import numpy as np 
import os

TF_CPP_MIN_LOG_LEVEL=2

from tensorflow.examples.tutorials.mnist import input_data

# Convolution Network Configurations
## Layer 1
filter_size_layer_1 = 5
num_filter_layer_1  = 16

## Layer 2
filter_size_layer_2 = 5
num_filter_layer_2  = 64

## Fully Connected Layer
num_of_fc = 128

batch_size = 64

def get_weight(layer_num, shape):

	W = tf.get_variable(name = 'W'+str(layer_num),
						shape = shape,
						dtype = tf.float32,
						initializer = tf.contrib.layers.xavier_initializer())

	return W

def get_bias(layer_num, shape):

	b = tf.Variable(initial_value = tf.zeros(shape),
					dtype = tf.float32,
					name = 'b' + str(layer_num))

	return b

def get_convolutional_layer(layer_num, filter_size, num_filter, input):

	num_channel = input.shape[3]
	w_shape = [filter_size, filter_size, num_channel, num_filter]
	b_shape = [num_filter]

	W = get_weight(layer_num, w_shape)
	b = get_bias(layer_num, b_shape)

	layer = tf.nn.conv2d(	input  = input,
							filter = W,
							strides= [1,1,1,1],
							padding= "SAME")
	layer += b

	layer = tf.nn.max_pool( value = layer,
							ksize = [1,2,2,1],
							strides = [1,1,1,1],
							padding = "SAME")

	layer = tf.nn.relu(layer)

	return layer

def get_fully_connected_layer(layer_num, input, output_size, use_relu=True):

	w_shape = [input.shape[1], output_size]
	b_shape = [output_size]

	W = get_weight(layer_num, w_shape)
	b = get_bias(layer_num, b_shape)
	
	y = tf.matmul(input, W) + b

	if use_relu:
		y = tf.nn.relu(y)

	return y

def main(restore):
	data = input_data.read_data_sets(os.path.join('data', 'MNIST'), one_hot=True)

	image_flat_size = data.train.images.shape[1]
	image_size = 28
	num_input_channel = 1
	num_classes = data.train.labels.shape[1]
	data_test_class = np.array([np.argmax(label) for label in data.test.labels])
	data_validation_class = np.array([np.argmax(label) for label in data.validation.labels])

	X = tf.placeholder( dtype = tf.float32,
						shape = [None, image_size, image_size, num_input_channel],
						name  = 'X')
	
	Y = tf.placeholder( dtype = tf.float32,
						shape = [None, num_classes],
						name  = 'Y')

	Y_true_class = tf.placeholder(  dtype = tf.int64,
									shape = [None],
									name = "Y_true_class")

	layer = get_convolutional_layer(layer_num = 1, 
									filter_size = filter_size_layer_1, 
									num_filter  = num_filter_layer_1, 
									input = X)

	layer = get_convolutional_layer(layer_num = 2, 
									filter_size = filter_size_layer_2, 
									num_filter  = num_filter_layer_2, 
									input = layer)
	layer_shape = layer.get_shape()
	flattened_layer_shape = layer_shape[1:4].num_elements()
	layer = tf.reshape(layer, [-1, flattened_layer_shape])

	layer = get_fully_connected_layer(  layer_num = 3,
										input = layer,
										output_size = num_of_fc)
	
	logits = get_fully_connected_layer( layer_num = 4,
										input = layer,
										output_size = num_classes)

	y_pred = tf.nn.softmax(logits)

	y_pred_class = tf.argmax(y_pred, axis =1)

	correct_predictions = tf.equal(y_pred_class, Y_true_class)
	accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

	saver = tf.train.Saver()

	save_dir = 'Checkpoints'

	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	save_path = os.path.join(save_dir, 'best_validation')


	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits,
															labels = Y)

	cost = tf.reduce_mean(cross_entropy)

	optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

	with tf.Session() as session:
		session.run(tf.global_variables_initializer())
		# full_pass = int(10)
		# global total_iterations
		best_validation_accuracy = -1
		last_improvement = 0

		for i in range(10000):
			x_batch , y_batch = data.train.next_batch(batch_size)
			x_batch = x_batch.reshape(x_batch.shape[0], image_size, image_size, 1)
			feed_dict = {X : x_batch, Y: y_batch}
			cost_val, _ = session.run([cost, optimizer], feed_dict = feed_dict)
			if i%100 == 0:
				print("Cost at {} is :\t{}".format(i,cost_val))
				x_validation, y_validation = data.validation.images, data.validation.labels
				x_validation = x_validation.reshape(x_validation.shape[0], image_size, image_size, 1)

				feed_valid_dict = {X: x_validation, Y: y_validation, Y_true_class: data_validation_class}
				acc = session.run(accuracy, feed_dict = feed_valid_dict)

				if best_validation_accuracy < acc:
					best_validation_accuracy = acc
					last_improvement = i
					saver.save(sess=session, save_path=save_path)
					print("Accuracy on Validation set is \t{}".format(acc))
			if i - last_improvement > 1000:
				break
		if(restore):
			saver.restore(sess=session, save_path=save_path)


		x_test = data.test.images.reshape(data.test.images.shape[0], image_size, image_size, 1)
		feed_dict_test = {X: x_test, Y: data.test.labels, Y_true_class: data_test_class}

		print("Accuracy for the test data set is :\t{}".format(session.run(accuracy, feed_dict= feed_dict_test)))


if __name__ == '__main__':
	main(restore=False)
