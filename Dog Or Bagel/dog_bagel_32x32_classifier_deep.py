import tensorflow as tf 
import dataset2 as ds
#Just an easy way to make a weight variable, with some random initialization
def weight_variable(shape):
	initial = tf.truncated_normal(shape,stddev=0.01);
	return tf.Variable(initial);
#Same idea
def bias_variable(shape):
	initial = tf.constant(0.1,shape=shape);
	return tf.Variable(initial);
#Zero padding, stride of size 1
def conv2d(x,W):
	return tf.nn.conv2d(x,W, strides=[1,1,1,1],padding='SAME');
#Pooling over 2x2 blocks
def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME');

#build tensorflow session THIS IS IMPORTANT
session = tf.InteractiveSession();

#1024 = 32x32 -> Images are 32x32. That's why I did this
x = tf.placeholder(tf.float32,shape=[None,3072]);
#Size 2 because right now we're just doing dog or bagel, that's it
y = tf.placeholder(tf.float32,shape=[None,8]);
#Our """""""""slope""""""""" (yes, this EXTREMELY oversimplifying it) weight
M = weight_variable([3072,8]);
#Our bias weight
b = bias_variable([8]);


'''
/////////////////////////////////////////////////////////////
Layer 1
/////////////////////////////////////////////////////////////

This is our first fancy shmancy convolutional layer. It's gonna
be pretty rad. Especially if it works
'''

'''
So, let me explain myself here
Right now we're going to take a __5x5__ kernel,
run it over the __3__ channel image (BGR)
and compute __32__ features.
Get it?
Good.
'''
M_conv1 = weight_variable([1,1,3,128]);

#Bias for our 32 features. Rad
b_conv1 = bias_variable([128]);

'''
Ok, so now we need to reshape our images into a 4d tensor. 
How? 
Watch

32x32, 3 channel (BGR) image. 
Now it's a 4d tensor...
We did it, reddit!
'''
x_image = tf.reshape(x,[-1,32,32,3])

'''
Now, we're going to convolve the image and
apply the ReLU function err whatever that does
'''
h_conv1 = tf.nn.relu(conv2d(x_image,M_conv1)+ b_conv1);

'''
Take the dog to the lake
'''
h_pool1 = max_pool_2x2(h_conv1);

'''
/////////////////////////////////////////////////////////////
Layer 2
/////////////////////////////////////////////////////////////

Pretty much the same idea, with some subtle differences
'''
M_conv2 = weight_variable([3,3,128,256]);
b_conv2 = bias_variable([256]);

h_conv2 = tf.nn.relu(conv2d(h_pool1,M_conv2)+ b_conv2);
h_pool2 = max_pool_2x2(h_conv2);


M_conv3 = weight_variable([2,2,256,512])
b_conv3 = bias_variable([512])

h_conv3 = tf.nn.relu(conv2d(h_pool2,M_conv3)+b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

M_conv4 = weight_variable([4,4,512,1024])
b_conv4 = bias_variable([1024])

h_conv4 = tf.nn.relu(conv2d(h_pool3,M_conv4)+b_conv4);
h_pool4 = max_pool_2x2(h_conv4)
'''
/////////////////////////////////////////////////////////////
Layer 3
/////////////////////////////////////////////////////////////

Now for our slightly less exciting, but still VERY important
fully connected layer. 

Without it, our fancy math blobs that we can thank our convolutional
layers for would mean basically nothing and we would be left with a
mess that we don't know how to deal with....

Remember that, kiddos.
'''

'''
So now we have our 'M' and 'b'
for our fully connected layer.

The size of the image was convolved by a
factor of 2, twice so (32/2)/2 = 8

and that's where I got that.

We have 64 features for a matrix of that size,
and we want to apply that to 1024 separate neurons....
hence the variables below
'''
M_fcl1 = weight_variable([4*1024,8192]);
b_fcl1 = bias_variable([8192]);

'''
So, now we need to flatten our pooled data from the previous
layer, hence the reshape call on h_pool2
'''
h_pool2_flat = tf.reshape(h_pool4,[-1,4*1024]);
'''
Then we make our hypothesis once again with
(hypothesis * weight) + bias
(x * M) + b
[back to my awful analogy]
'''
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,M_fcl1)+b_fcl1);


'''
Now we're going to apply dropout to hopefully make
things better
'''
keep_prob = tf.placeholder(tf.float32);
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob);




'''
/////////////////////////////////////////////////////////////
Layer 4
/////////////////////////////////////////////////////////////

This is our classification layer. The good one.
The one that gives us the answer. 
Yeah.
Cool.
(once again, this is a fully connected layer)
'''

M_fcl2 = weight_variable([8192,8]);
b_fcl2 = bias_variable([8]);
#Now, we're applying a softmax function to get our final answers instead of ReLU
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,M_fcl2)+b_fcl2);


'''
-------------------------------------|
-------------------------------------|
TRAINING ZONE
-------------------------------------|
-------------------------------------|

and now, a message from our sponsors:

The FitnessGram Pacer Test is a multistage aerobic capacity test that progressively gets 
more difficult as it continues. The 20 meter pacer test will begin in 30 seconds. Line up 
at the start. The running speed starts slowly, but gets faster each minute after you hear 
this signal. [beep] A single lap should be completed each time you hear this sound. [ding] 
Remember to run in a straight line, and run as long as possible. The second time you fail 
to complete a lap before the sound, your test is over. The test will begin on the word start. 
On your mark, get ready, start.

ANYWAYS:
Great! Now we've completed the structure of our neural network
Now we have to train our neural network (and pray it works)

Luckily (like everything else) google has made it relatively simple
for us to train our ConvNN in tensorflow

So here goes
'''

#Building my dataset
print("Gathering Dataset...")
learn_rate = 1e-8
train_data = ds.get_train_dataset(start=10,end=3000);
test_data = ds.get_test_dataset();
size = len(train_data[0])
batch_size = 250
check_freq = 100
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y_conv,y);
train_step = tf.train.AdamOptimizer(learn_rate).minimize(cross_entropy);
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y,1));
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32));
session.run(tf.initialize_all_variables());

print("Beginning training...")
acc = 0;
test_dict = {
	x:test_data[0],
	y:test_data[1],
	keep_prob:1.0
}

for i in range(10000):
	train_eval_dict = {
		x:train_data[0][i*batch_size%size:(i+1)*batch_size%size],
		y:train_data[1][i*batch_size%size:(i+1)*batch_size%size],
		keep_prob:1.0	
	}
	train_dict = {
		x:train_data[0][i*batch_size%size:(i+1)*batch_size%size],
		y:train_data[1][i*batch_size%size:(i+1)*batch_size%size],
		keep_prob:0.5	
	}
	if i%check_freq == 0:
		train_accuracy = accuracy.eval(feed_dict=train_eval_dict);
		test_accuracy = accuracy.eval(feed_dict=test_dict)
		print("Step %d\ttraining accuracy: %g\ttest accuracy: %g"%(i,train_accuracy,test_accuracy));
		#if train_accuracy > acc:
			#acc = train_accuracy;
		#elif train_accuracy < acc:
			#if learn_rate/3 >0:
				#learn_rate /= 10;
				#print("New learn rate: %e"%(learn_rate));
		#else:
			#learn_rate *=5;
			#print("New learn rate: %e"%(learn_rate));
		print y_conv.eval(feed_dict=train_dict)
	train_step.run(feed_dict=train_dict);
	
print("Final test accuracy %g"%accuracy.eval(feed_dict=test_dict));
print("End")
session.close()
