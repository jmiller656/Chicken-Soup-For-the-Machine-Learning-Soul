import tensorflow as tf
import convnet as cnn
net = cnn.CNN(size = 100)
learn_rate = 1e-8
train_data = #todo
test_data = #todo
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(net['y_conv'],net['y'])
train_step = tf.train.AdamOptimizer(learn_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(net['y_conv'],1),tf.argmax(net['y'],1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction))
session.run(tf.initialize_all_variables())
print("Beginning training...")
for i in range(1000):
	if i%100 == 0:
		train_accuracy = accuracy.eval(feed_dict={x:test_data[0],y:test_data[1],keep_prob:1.0})
		print("Step %d training accuracy %g"%(i,train_accuracy))
	train_step.run(feed_dict={x:train_data[0],y:train_data[1],keep_prob:0.5})

print("Final test accuracy: %g"%(accuracy.eval(feed_dict={x:test_data[0],y:test_data[1],keep_prob:1.0})))
