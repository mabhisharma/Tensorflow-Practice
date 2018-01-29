import tensorflow as tf 
TF_CPP_MIN_LOG_LEVEL=2
import numpy as np 
import os
from tensorflow.examples.tutorials.mnist import input_data

# Wrapper and Helper functions -

def get_weight(layer_num, shape):
	
	return tf.get_variable(	name='W'+str(layer_num),
							shape=shape,
							dtype=tf.float32,
							initializer=tf.contrib.layers.xavier_initializer())

def get_bias(layer_num, shape):

	return tf.Variable(	initial_value=tf.zeros(shape), 
					dtype=tf.float32,
					name='b'+str(layer_num))

def get_convolutional_layer(layer_num, input, num_filter, filter_size, num_channel):

	W = get_weight(layer_num, shape=[filter_size, filter_size, num_channel, num_filter])
	b = get_bias(layer_num, shape=[num_filter])

	layer = tf.nn.conv2d(	input = input,
							filter=W,
							strides=[1,1,1,1],
							padding="SAME"	)
	layer += b

	layer = tf.nn.max_pool(	value=layer,
							ksize=[1,2,2,1],
							strides=[1,2,2,1],
							padding="SAME"	)

	layer = tf.nn.relu(layer)

	return layer	

def get_fully_connected_layer(layer_num, input, weight_in_dim, weight_out_dim):
	W = get_weight(layer_num, [weight_in_dim, weight_out_dim])
	b = get_bias(layer_num, [weight_out_dim])

	y = tf.matmul(input, W) + b
	
	return y



# Load the data from MNIST data set
data = input_data.read_data_sets(os.path.join("data", "MNIST"), one_hot=True)

num_of_classes = data.train.labels.shape[1]
print("\nTotal number of output Classes is :\t{}".format(num_of_classes))

image_flat_shape = data.train.images.shape[1]
image_size = 28
print("Image Shape is :\t{}".format(image_flat_shape))

data.test.cls = np.array([label.argmax() for label in data.test.labels])

# ## Print the size of dataset
# print("Size of -")
# print("- Training set is :\t{}".format(len(data.train.labels)))
# print("- Testing set is :\t{}".format(len(data.test.labels)))

# Configuration of the Convolutional network
## Conv Layer 1
filter_1_size = 5
num_of_filter_in_1 = 16

## Conv Layer 2
filter_2_size = 5
num_of_filter_in_2 = 64

## Fully Connected Layer
num_of_fc = 128

batch_size = 16

# Create Placeholders and Variables
X = tf.placeholder( dtype=tf.float32, 
					shape=[None, image_size, image_size, 1],
					name='X')

Y = tf.placeholder( dtype=tf.float32,
					shape=[None, num_of_classes],
					name='Y')

Y_true_class = tf.placeholder( dtype=tf.int64,
					shape=[None],
					name='Y_true_class')

# Model
layer = get_convolutional_layer(1, X, num_of_filter_in_1, filter_1_size, 1)
layer = get_convolutional_layer(2, layer, num_of_filter_in_2, filter_2_size, num_of_filter_in_1)
layer_shape = layer.get_shape()
num_features = layer_shape[1:4].num_elements()
layer_flat = tf.reshape(layer, [-1, num_features])
layer = get_fully_connected_layer(3, layer_flat, num_features, num_of_fc)
layer = tf.nn.relu(layer)
logits = get_fully_connected_layer(4, layer, num_of_fc, num_of_classes)
y_pred = tf.nn.softmax(logits)
y_pred_class = tf.argmax(logits, axis = 1)

# Cost
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits= logits,
														labels= Y)

cost = tf.reduce_mean(cross_entropy)

correct_predictions = tf.equal(y_pred_class, Y_true_class)
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

# Optimization
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

# Create Session
with tf.Session() as session:
	session.run(tf.global_variables_initializer())
	for i in range(10000):
		x_batch , y_batch = data.train.next_batch(batch_size)
		x_batch = x_batch.reshape(x_batch.shape[0], image_size, image_size, 1)
		cost_val, _ = session.run([cost, optimizer], feed_dict={X: x_batch, Y: y_batch})
		if i%10 == 0:
			print("Cost at {} is :\t{}".format(i,cost_val))

	x_test = data.test.images.reshape(data.test.images.shape[0], image_size, image_size, 1)
	feed_dict_test = {X: x_test, Y: data.test.labels, Y_true_class: data.test.cls}

	print("Accuracy for the test data set is :\t{}".format(session.run(accuracy, feed_dict= feed_dict_test)))


