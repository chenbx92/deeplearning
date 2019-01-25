import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def add_FClayer(inputs,insize,outsize,name,activation_function=None):
    with tf.name_scope('FClayer'+str(name)):
        with tf.name_scope('weights'):
            W=tf.Variable(tf.random_normal([insize,outsize]))
        with tf.name_scope('biases'):
            b=tf.Variable(tf.zeros([1,outsize])+0.1)
        with tf.name_scope('outflow'):
            outflow=tf.matmul(inputs,W)+b
    if activation_function==None:
        outputs=outflow
    else:
        outputs=activation_function(outflow)
    return outputs

x_data=np.float32(np.linspace(-1,1,300)[:,np.newaxis])
noise=np.random.normal(0,0.05,x_data.shape)
y_data=np.square(x_data)-0.5+noise

with tf.name_scope('inputsx'):
    xs=tf.placeholder(tf.float32)
with tf.name_scope('inputslabel'):
    ys=tf.placeholder(tf.float32)

l1=add_FClayer(xs,1,10,'1',activation_function=tf.nn.relu)
prediction=add_FClayer(l1,10,1,'2',activation_function=None)

with tf.name_scope('loss'):
    loss=tf.reduce_mean(tf.square(ys-prediction))


train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
sess = tf.Session()
init = tf.global_variables_initializer()
writer=tf.summary.FileWriter("logs",sess.graph)
sess.run(init)


for i in range(300):

   sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
   if i % 100 == 0:
       print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))

sess.close()