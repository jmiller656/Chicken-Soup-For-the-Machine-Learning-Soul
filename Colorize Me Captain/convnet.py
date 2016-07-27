import tensorflow as tf

def weight_variable(shape):
	initial = tf.truncated_normal(shape,stddev=0.01);
	return tf.Variable(initial);
def bias_variable(shape):
	initial = tf.constant(0.1,shape=shape);
	return tf.Variable(initial);
def conv2d(x,W):
	return tf.nn.conv2d(x,W, strides=[1,1,1,1],padding='SAME');
def max_pool(x,stride=2):
	return tf.nn.max_pool(x,ksize=[1,stride,stride,1],strides=[1,stride,stride,1],padding='SAME');

def CNN(inputsize = 32):
	size_squared = inputsize*inputsize
	output_size = size_squared*3
	x = tf.placeholder(tf.float32,shape=[None,size_squared])
	y = tf.placeholder(tf.float32,shape=[None,output_size])
	M = weight_variable([size_squared,output_size])
	b = bias_variable(output_size)
	#Conv Layer 1
	M_conv1 = weight_variable([5,5,1,output_size])
	b_conv1 = bias_variable([output_size])
	x_image = tf.reshape(x,[-1,inputsize,inputsize,1])
	h_conv1 = tf.nn.relu(conv2d(x_image,M_conv1),+b_conv1)
	h_pool1 = max_pool(h_conv1,stride=5)
	#Conv Layer 2
	M_conv2 = weight_variable([3,3,output_size,output_size*4])
	b_conv2 = bias_variable([output_size*4])
	h_conv2 = tf.nn.relu(conv2d(h_pool1,M_conv2)+b_conv2)
	h_pool2 = max_pool(h_conv2)
	#Fully Connected Layer 1
	M_fcl1 = weight_variable([size/4*size/4*output_size*4,output_size*2])
	b_fcl1 = bias_variable([output_size*2])
	h_pool_flat = tf.reshape(h_pool2,[-1,size/4*size*output_size])
	h_fcl1 = tf.nn.relu(tf.matmul(h_pool_flat,M_fcl1)+b_fcl1)
	keep_prob = tf.placeholder(tf.float32)
	h_fcl1_drop = tf.nn.dropout(h_fcl1,keep_prob)
	#Fully Connected Layer 2
	M_fcl2 = weight_variable([output_size*2,output_size])
	b_fcl2 = bias_variable(output_size)
	y_conv = tf.nn.softmax(tf.matmul(h_fcl1_drop,M_fcl2)+b_fcl2)
	finalCNN = {
		'y' = y,
		'y_conv' = y_conv,
	}
	return finalCNN
