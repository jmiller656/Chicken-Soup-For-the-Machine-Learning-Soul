import tensorflow as tf

def weight_variable(shape):
	initial = tf.truncated_normal(shape,stddev=0.01)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1,shape=shape)
	return tf.Variable(initial)

def conv2d(x,W):
	return tf.nn.conv2d(x,W, strides=[1,1,1,1],padding='SAME')

def max_pool(x,dim=2):
	return tf.nn.max_pool(x,ksize=[1,dim,dim,1],strides=[1,dim,dim,1],padding='SAME')

class CNN:

	def convLayer(self,input_tensor,features,dimension=1,channels=1,max_pool=True):
		M_conv = weight_variable([dimension,dimension,channels,features],name='M_conv' + str(self.convlayers))
		b_conv = bias_variable([features],name ='b_conv' + str(self.convlayers))
		h_conv = tf.nn.relu(conv2d(input_tensor,M_conv)+b_conv)
		tensors.update({
			'M_conv'+str(self.convlayers): M_conv,
			'b_conv'+str(self.convlayers): b_conv,
			'h_conv'+str(self.convlayers): h_conv
		})
		if max_pool:
			h_pool = max_pool(h_conv)
			return h_pool
		return h_conv

	def fullyConnectedLayer(self,input_tensor,features,activation=''):
		M_fcl = weight_variable([int(input_tensor.get_shape()[1]),features],name="M_fcl"+str(self.fclayers))
		b_fcl = bias_variable([features],name="b_fcl"+str(self.fclayers))
		h_fcl = None
		if activation == 'relu':
			h_fcl = tf.nn.relu(tf.matmul(input_tensor,M_fcl)+b_fcl)
		elif activation == 'softmax':
			h_fcl = tf.nn.softmax(tf.matmul(input_tensor,M_fcl)+b_fcl)
		else:
			h_fcl = tf.nn.sigmoid(tf.matmul(input_tensor,M_fcl)+b_fcl)
		self.tensors.update({
			'M_fcl' + str(self.fclayers): M_fcl,
			'b_fcl' + str(self.fclayers): b_fcl,
			'h_fcl' + str(self.fclayers): h_fcl
		})
		return h_fcl
	
	def __init__(self,inputSize,outputSize,rgb=False,dtype = tf.float32):
		self.inputSize = inputSize
		self.outputSize = outputSize
		self.tensors = {}
		self.fclayers = 0
		self.convlayers = 0
		channels =  3 if rgb else 1
		#Create input and output variables, add to dict		
		x = tf.placeholder(dtype,shape=[None,inputSize*inputSize*channels])
		self.tensors.update({'x':x})
		y = tf.placeholder(dtype,shape=[None,outputSize])
		self.tensors.update({'y':y})
		#Create initial weights and biases, add to dict
		M = weight_variable([inputSize*inputSize*channels,outputSize])
		b = bias_variable([outputSize])
		self.tensors.update({'M':M})
		self.tensors.update({'b':b})
		#Create first convolutional layer
		conv_weights = [128,256]
		M_conv1 = weight_variable([1,1,channels,conv_weights[0]])
		b_conv1 = bias_variable([conv_weights[0]])
		x_image = tf.reshape(x,[-1,inputSize,inputSize,channels])
		self.tensors.update({'M_conv1':M_conv1})
		self.tensors.update({'b_conv1':b_conv1})
		#Make first conv hypothesis
		h_conv1 = tf.nn.relu(conv2d(x_image,M_conv1)+ b_conv1)
		h_pool1 = max_pool(h_conv1)
		self.tensors.update({'h_conv1':h_conv1})
		self.tensors.update({'h_pool1':h_pool1})
		#Create second convolutional layer
		M_conv2 = weight_variable([1,1,conv_weights[0],conv_weights[1]])
		b_conv2 = bias_variable([conv_weights[1]])
		self.tensors.update({'M_conv2':M_conv2})
		self.tensors.update({'b_conv2':b_conv2})
		h_conv2 = tf.nn.relu(conv2d(h_pool1,M_conv2)+b_conv2)
		h_pool2 = max_pool(h_conv2)
		self.tensors.update({'h_conv2':h_conv2})
		self.tensors.update({'h_pool2':h_pool2})
		pool2shape = h_pool2.get_shape()
		size = int(pool2shape[1] * pool2shape[2] * pool2shape[3])
		h_pool2_flat = tf.reshape(h_pool2,[-1,size])
		#First Fully Connected layer
		M_fcl1 = weight_variable([size,1024])
		b_fcl1 = bias_variable([1024])
		self.tensors.update({'M_fcl1': M_fcl1})
		self.tensors.update({'b_fcl1': b_fcl1})
		h_fcl1 = tf.nn.relu(tf.matmul(h_pool2_flat,M_fcl1)+b_fcl1)
		self.tensors.update({'h_fcl1':h_fcl1})
		#Applying drouput
		keep_prob = tf.placeholder(tf.float32,name='keep_prob')
		self.tensors.update({'keep_prob':keep_prob})
		h_fcl1_drop = tf.nn.dropout(h_fcl1,keep_prob)
		#Fully connected layer 2
		M_fcl2 = weight_variable([1024,outputSize])
		b_fcl2 = bias_variable([outputSize])
		self.tensors.update({'M_fcl2':M_fcl2})
		self.tensors.update({'b_fcl2':b_fcl2})
		y_conv = tf.nn.softmax(tf.matmul(h_fcl1_drop,M_fcl2)+b_fcl2)
		self.tensors.update({'y_conv':y_conv})
		
