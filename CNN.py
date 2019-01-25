import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

def add_convlayer(inputs,Wshape,name,activation_function=None):
    with tf.name_scope(str(name)):
        with tf.name_scope('weights'):
            W=tf.Variable(tf.truncated_normal(Wshape,stddev=0.1))
        with tf.name_scope('biases'):
            b=tf.Variable(tf.constant(0.1,shape=[Wshape[3]]))
        with tf.name_scope('outflow'):
            outflow=tf.nn.conv2d(inputs,W,strides=[1,1,1,1],padding='SAME')
    if activation_function==None:
        outputs=outflow+b
    else:
        outputs=activation_function(outflow+b)
    return outputs

def add_poolinglayer(inputs,name):
    with tf.name_scope(str(name)):
        with tf.name_scope('outflow'):
            outflow=tf.nn.max_pool(inputs,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    return outflow

def add_FClayer(inputs,insize,outsize,name,keep_prob,activation_function=None):
    with tf.name_scope('FClayer'+str(name)):
        with tf.name_scope('weights'):
            W=tf.Variable(tf.random_normal([insize,outsize]))
        with tf.name_scope('biases'):
            b=tf.Variable(tf.zeros([1,outsize])+0.1)
        with tf.name_scope('outflow'):
            outflow=tf.matmul(inputs,W)+b
    if activation_function==None:
        outputs=tf.nn.dropout(outflow,keep_prob)

    else:
        outputs=tf.nn.dropout(activation_function(outflow),keep_prob)
    return outputs




#inputs
with tf.name_scope('x'):
    xs=tf.placeholder(tf.float32,[None,784])
with tf.name_scope('y'):
    ys=tf.placeholder(tf.float32,[None,10])
keep_prob=tf.placeholder(tf.float32)
x_data=tf.reshape(xs,[-1,28,28,1])

#conv layer 1
convlayer1=add_convlayer(x_data,[5,5,1,32],'convlayer1',tf.nn.relu) #28*28*32
#pooling layer 1
poolinglayer1=add_poolinglayer(convlayer1,'poolinglayer1') #14*14*32

#conv layer 2
convlayer2=add_convlayer(poolinglayer1,[5,5,32,64],'convlayer2',tf.nn.relu) #14*14*64
#pooling layer 2
poolinglayer2=add_poolinglayer(convlayer2,'poolinglayer2') #7*7*64

FCinputs=tf.reshape(poolinglayer2,[-1,7*7*64])
#FC layer1
FClayer=add_FClayer(FCinputs,7*7*64,1024,'FClayer1',keep_prob,tf.nn.relu)

#outputlayyer
prediction=add_FClayer(FClayer,1024,10,'outputlayer',1,tf.nn.softmax)

with tf.name_scope('loss'):
    loss=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
train=tf.train.GradientDescentOptimizer(0.001).minimize(loss)

sess=tf.Session()
sess.run(tf.global_variables_initializer())
writer=tf.summary.FileWriter("logs",sess.graph)
writer.close()

for i in range(1000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    sess.run(train,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.5})
    if i%50==0:
        y_pre=sess.run(prediction,feed_dict={xs:mnist.test.images,keep_prob:1})
        correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(mnist.test.labels,1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        print(sess.run(accuracy))
        print(sess.run(loss,feed_dict={xs:mnist.test.images,ys:mnist.test.labels,keep_prob:1}))


sess.close()