
# coding: utf-8

# # Neural Network Example
# 
# Build a 2-hidden layers fully connected neural network (a.k.a multilayer perceptron) with TensorFlow.
# 
# - Author: Aymeric Damien
# - Project: https://github.com/aymericdamien/TensorFlow-Examples/

# ## Neural Network Overview
# 
# <img src="http://cs231n.github.io/assets/nn1/neural_net2.jpeg" alt="nn" style="width: 400px;"/>
# 
# ## MNIST Dataset Overview
# 
# This example is using MNIST handwritten digits. The dataset contains 60,000 examples for training and 10,000 examples for testing. The digits have been size-normalized and centered in a fixed-size image (28x28 pixels) with values from 0 to 1. For simplicity, each image has been flattened and converted to a 1-D numpy array of 784 features (28*28).
# 
# ![MNIST Dataset](http://neuralnetworksanddeeplearning.com/images/mnist_100_digits.png)
# 
# More info: http://yann.lecun.com/exdb/mnist/

# In[1]:


from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
import tensorflow as tf
import timeit
# In[17]:

#accuracies = []
#for num_hidden_1 in range(700, 50, -5):
# Parameters
learning_rate = 0.01
num_steps = 20000
batch_size = 5000
display_step = 100

# Network Parameters
#n_hidden_1 = num_hidden_1  # 1st layer number of neurons
n_hidden_1 = 10  # 1st layer number of neurons
n_hidden_2 = 10  # 2nd layer number of neurons
num_input = 784  # MNIST data input (img shape: 28*28)
num_classes = 10  # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])


# In[18]:


# Store layers weight & bias
weights = {
	'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1]), name="layer1"),
	#'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name="layer2"),
	#'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_2]), name="layer3"),  # just me playing around with the architecture
	'out': tf.Variable(tf.random_normal([n_hidden_1, num_classes]), name="outlayer")
}
biases = {
	'b1': tf.Variable(tf.random_normal([n_hidden_1]), name="layer1_bias"),
	#'b2': tf.Variable(tf.random_normal([n_hidden_2]), name="layer2_bias"),
	#'b3': tf.Variable(tf.random_normal([n_hidden_2]), name="layer3_bias"),
	'out': tf.Variable(tf.random_normal([num_classes]), name="outlayer_bias")
}


# In[19]:


# Create model
def neural_net(x):
	# Hidden fully connected layer with 256 neurons
	layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
	# Hidden fully connected layer with 256 neurons
	#layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
	#layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
	# Output fully connected layer with a neuron for each class
	#out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
	out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
	return out_layer


# In[20]:


# Construct model
logits = neural_net(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
#optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


# In[21]:

t0 = timeit.default_timer()
# Start training
sess = tf.Session()
# Run the initializer
sess.run(init)
# save the log file for tensorboard
writer = tf.summary.FileWriter("nn_raw_log", sess.graph)


for step in range(1, num_steps+1):
	batch_x, batch_y = mnist.train.next_batch(batch_size)
	# Run optimization op (backprop)
	sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
	if step % display_step == 0 or step == 1:
		# Calculate batch loss and accuracy
		loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
		print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " + "{:.3f}".format(acc))

#print("Optimization Finished!")

# Calculate accuracy for MNIST test images
print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
t1 = timeit.default_timer()
print(t1-t0)
#accuracies.append(sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))








#########################################
# Take a look at the actual data
import matplotlib.pyplot as plt
plt.imshow(batch_x[1, :].reshape((28, 28)))  # batch_x is a big matrix with batch_size rows and num_input columns = 28x28
plt.show()

#########################################
# Let's take a look at the ones it got wrong
import numpy as np
import time
is_correct = sess.run(correct_pred, feed_dict={X: batch_x, Y: batch_y})
prediction = sess.run(tf.argmax(logits, 1), feed_dict={X: batch_x, Y: batch_y})
actual = sess.run(tf.argmax(Y, 1), feed_dict={X: batch_x, Y: batch_y})
for bad_loc in np.where(is_correct == False)[0]:
	plt.imshow(batch_x[bad_loc, :].reshape((28, 28)), cmap='gray')
	plt.title("Actual %s, predicted %s" % (actual[bad_loc], prediction[bad_loc]))
	plt.show()
	time.sleep(2)
