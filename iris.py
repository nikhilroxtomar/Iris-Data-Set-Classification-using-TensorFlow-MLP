
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'

import tensorflow as tf
import numpy as np
import time

start_time = time.time()

def label_encode(label):
	val=[]
	if label == "Iris-setosa":
		val = [1,0,0]
	elif label == "Iris-versicolor":
		val = [0,1,0]
	elif label == "Iris-virginica":
		val = [0,0,1]	
	return val

def data_encode(file):
	X = []
	Y = []
	train_file = open(file, 'r')
	for line in train_file.read().strip().split('\n'):
		line = line.split(',')
		X.append([line[0], line[1], line[2], line[3]])
		Y.append(label_encode(line[4]))
	return X, Y

#Defining a Multilayer Perceptron Model
def model(x, weights, bias):
	layer_1 = tf.add(tf.matmul(x, weights["hidden"]), bias["hidden"])
	layer_1 = tf.nn.relu(layer_1)

	output_layer = tf.matmul(layer_1, weights["output"]) + bias["output"]
	return output_layer

#Training and Testing Data
train_X , train_Y = data_encode('iris.train')
test_X , test_Y = data_encode('iris.test')


#hyperparameter
learning_rate = 0.01
training_epochs = 2000
display_steps = 200


#Network parameters
n_input = 4
n_hidden = 10
n_output = 3

#Graph Nodes
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])
		
#Weights and Biases
weights = {
	"hidden" : tf.Variable(tf.random_normal([n_input, n_hidden]), name="weight_hidden"),
	"output" : tf.Variable(tf.random_normal([n_hidden, n_output]), name="weight_output")
}

bias = {
	"hidden" : tf.Variable(tf.random_normal([n_hidden]), name="bias_hidden"),
	"output" : tf.Variable(tf.random_normal([n_output]), name="bias_output")
}	

#Define model
pred = model(X, weights, bias) 

#Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#Initializing global variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	for epoch in range(training_epochs):
		_, c = sess.run([optimizer, cost], feed_dict={X: train_X, Y: train_Y})
		if(epoch + 1) % display_steps == 0:
			print "Epoch: ", (epoch+1), "Cost: ", c
	
	print("Optimization Finished!")
	
	test_result = sess.run(pred, feed_dict={X: train_X})
	correct_pred = tf.equal(tf.argmax(test_result, 1), tf.argmax(train_Y, 1))

	accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"))
	print "Accuracy:", accuracy.eval({X: test_X, Y: test_Y})


end_time = time.time()

print "Completed in ", end_time - start_time , " seconds"

	


