'''
Josh Miller
6/27/16
Convolutional Neural Net tutorial from tensorflow.org
This project was built with guidance from the tutorial found at the following address:
https://www.tensorflow.org/versions/r0.9/tutorials/mnist/pros/index.html

Moral of the story in simple terms:
Neural nets are like ogres... which are like onions... which have layers
'''
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf 

#Added a small amount of noise for symmetry breaking and mitigate the effect of "dead neurons"
def weight_variable(shape):
	initial = tf.truncated_normal(shape,stddev=0.1);
	return tf.Variable(initial);
#Same idea
def bias_variable(shape):
	initial = tf.constant(0.1,shape=shape);
	return tf.Variable(initial);
#Zero padding, stride of size 1
def conv2d(x,W):
	return tf.nn.conv2d(x,W, strides=[1,1,1,1],padding='SAME');
#Pooling occours over 2x2 blocks
def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME');

#Getting data, and creating tensorflow session
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True);
session = tf.InteractiveSession();
#also building initial variables
x = tf.placeholder(tf.float32,shape=[None,784]);
y_  = tf.placeholder(tf.float32,shape=[None,10]);
W = tf.Variable(tf.zeros([784,10]));
b = tf.Variable(tf.zeros([10]));
'''
////////////////////
BEGINNING OF LAYER 1
////////////////////
'''
'''
First two dimensions are patch size (5x5)
Next is the number of input channels
Followed by the number of output channels
Therefore, we compute 32 features for a 5x5 patch
'''
W_conv1 = weight_variable([5,5,1,32]);

#Bias vector with a component for each output channel
b_conv1 = bias_variable([32]);

'''
Reshape x image into a 4d tensor
dimension 1: I have no idea tbh
dimension 2&3: dimensions of image
dimension 4: number of color channels in the image
'''
x_image = tf.reshape(x,[-1,28,28,1]);

'''
Now, we're gonna convolve the image with the weight tensor, 
add the bias to it, and apply the ReLU function
....... but Josh......
what is the ReLU function?
Good question! Here's a good place to read up on it:
https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
'''
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1);

'''
Afterwards, we're going to max pool
Here's a good informative section on max pooling:
http://i.imgur.com/s6JsTzk.gifv
..................................
just kidding, here's a better one:
https://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer
'''
h_pool1 = max_pool_2x2(h_conv1);

'''
////////////////////
END OF LAYER 1
////////////////////
'''

'''
////////////////////
BEGINNING OF LAYER 2
////////////////////
'''

'''
Alright, so here's the deal:
this is a deep network, so we have to stack several layers
like this. Make sense?

This time, we're going to have 64 features for a 5x5 patch
'''
W_conv2 = weight_variable([5,5,32,64]);
b_conv2 = bias_variable([64]);

h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2);
h_pool2 = max_pool_2x2(h_conv2);

'''
////////////////////
END OF LAYER 2
////////////////////
'''

'''
////////////////////
BEGINNING OF LAYER 3
////////////////////
'''

'''
So, apparently the image size has been reduced
to 7x7 now (not quite sure how...)

anyways, now what we're going to do is add a fully
connected layer with 1024 neurons, which will allow
processing on the entire image. Then, we'll reshape
the tensor from the pooling layer into a group of vectors,
multiply by a weight matrix, add a bias and apply our
good ol' friend, the ReLU

*whew*
'''
W_fc1 = weight_variable([7*7*64,1024]);
b_fc1 = bias_variable([1024]);

h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64]);
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1);
'''
////////////////////
END OF LAYER 3
////////////////////
'''

'''
Now, there's a problem that comes with all the complex
spookymaths when you have a large number of features.
It is recognized by the name <<<OVERFITTING>>>
Spooky...

Anyways, what we're going to do next is deploy a measure that we
will take in order to mitigate this problem (cool!)

Our strategy for doing so is called Droupout.

<Ignore this, it doesn't have anything to do with machine learning>
No, I'm not referring to that one weird kid you who left your high school
at 16 to pursue some kind of genius venture that never really panned out,
constantly chanting the refrain "BILL GATES AND STEVE JOBS WERE DROUPOUTS, LOOK HOW SUCCESSFUL THEY WERE"

Well guess what:
1. Those men didn't drop out of high school, they dropped out of VERY PRESTIGIOUS UNIVERSITIES
2. They didn't leave to pursue a promising career of overtime at mcdonalds in order to feed their drug habit, or what have you; 
   they actually wanted to change the world... and they did

WHEW!
</ok, my silly rant is over now, you're good... I promise>

What dropout does in the context of machine learning is, essentially
thin the network and remove neurons that don't seem quite helpful (or something like that)
Here's a link with a much more formal description:
https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf

Now luckily, tensorflow is very nice and automatically handles scaling
neuron outputs in addition to masking them, so it works without any additional
scaling. Thanks Tensorflow!
'''
keep_prob = tf.placeholder(tf.float32);
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob);

'''
////////////////////
BEGINNING OF LAYER 4
////////////////////
'''

'''
This is the readout layer of the neuralnet.
Basically, what we do is add a softmax layer, just like
the previous, much more simple example 
'''

W_fc2 = weight_variable([1024,10]);
b_fc2 = bias_variable([10]);

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2);

'''
////////////////////
END OF LAYER 4
////////////////////
'''

'''
Welp.
Structurally, we have the entire neural net. Now, for the
exciting part! (jk, I honestly find all of this to be quite exciting,
don't you?)
Anyways, from here on we will be training the neural net and
tracking how effective it is. Cool!
'''
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv),reduction_indices=[1]));
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy);
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1));
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32));
session.run(tf.initialize_all_variables());
for i in range(20000):
	batch = mnist.train.next_batch(50);
	if i%100 == 0:
		train_accuracy = accuracy.eval(feed_dict={
			x:batch[0],y_:batch[1],keep_prob:1.0});
		print("step %d, training accuracy %g"%(i,train_accuracy));
	train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5});

print("Final test accuracy %g"%accuracy.eval(feed_dict={
	x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}));