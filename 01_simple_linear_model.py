import tensorflow as tf 
import numpy as np 
import os

from tensorflow.examples.tutorials.mnist import input_data


data = input_data.read_data_sets(os.path.join("data", "MNIST"), one_hot=True)

print("Size of -")
print("- Training set is\t {}".format(len(data.train.labels)))
print("- Testing set is\t {}".format(len(data.test.labels)))

batch_size = 100
image_size = 28
image_shape = (28,28)
image_flat_size = image_size*image_size
num_classes = data.train.labels.shape[1]

data.test.cls = np.array([label.argmax() for label in data.test.labels])
X = tf.placeholder(dtype=tf.float32, shape=[None, image_flat_size], name='X')
Y = tf.placeholder(dtype=tf.float32, shape=[None, num_classes], name='Y')
y_true_cls = tf.placeholder(dtype=tf.int64, shape=[None,], name='y_true_cls')


W = tf.get_variable(name='W', 
					shape=[image_flat_size, num_classes], 
					dtype=tf.float32, 
					initializer=tf.contrib.layers.xavier_initializer())

b = tf.Variable(initial_value=tf.zeros([num_classes]), name='b', dtype=tf.float32)

#Model
logits = tf.matmul(X, W) + b

y_pred = tf.nn.softmax(logits)

y_pred_cls = tf.argmax(y_pred, axis = 1)

correct_predictions =  tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


#CostFunction

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)

cost = tf.reduce_mean(cross_entropy)

#Optimizer

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)

#Session

session = tf.Session()
session.run(tf.global_variables_initializer())

def optimization(session):
	for i in range(1000):
		x_batch , y_batch = data.train.next_batch(batch_size)
		cost_val, _ = session.run([cost, optimizer], feed_dict={X: x_batch, Y: y_batch})
		# if i%10 == 0:
		# 	print("Cost at {} is :\t{}".format(i,cost_val))

optimization(session)

feed_dict_test = {X: data.test.images, Y: data.test.labels, y_true_cls: data.test.cls}

print("Accuracy for the test data set is :\t{}".format(session.run(accuracy, feed_dict= feed_dict_test)))

session.close()




