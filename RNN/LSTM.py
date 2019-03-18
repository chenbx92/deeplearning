import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

with tf.name_scope("input_x"):
    x=tf.placeholder(tf.float32,[None,28,28],name="input_x")
tf.add_to_collection("inputx",x)
with tf.name_scope("input_y"):
    y=tf.placeholder(tf.float32,[None,10],name="input_y")

weights={
    'in': tf.Variable(tf.random_normal([28,128]),name="Weightsin"),
    'out': tf.Variable(tf.random_normal([128,10]),name="Weightsout")
}
biases={
    'in': tf.Variable(tf.constant(0.1,shape=[1,128]),name="biasin"),
    'out': tf.Variable(tf.constant(0.1,shape=[1,10]),name="biasout")
}


def LSTM(X,weights,biases):
    with tf.name_scope("inlayer"):
        X=tf.reshape(X,[-1,28])
        X_in=tf.matmul(X,weights['in'])+biases['in']

    X_in=tf.reshape(X_in,[-1,28,128]) #batches,steps,inputs

    with tf.name_scope("lstmlayer"):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(128,forget_bias=1.0,state_is_tuple=True) #lstm cells number
        init_state = lstm_cell.zero_state(100,dtype=tf.float32) #batch size

        outputs,final_state=tf.nn.dynamic_rnn(lstm_cell,X_in,initial_state=init_state,time_major=False)

    with tf.name_scope("outlayer"):
        outputs=tf.unstack(tf.transpose(outputs,[1,0,2]))
        out=tf.matmul(outputs[-1],weights['out'])+biases['out']

    return out


pred=LSTM(x,weights,biases)
tf.add_to_collection("predict",pred)

with tf.name_scope("loss"):
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))

train=tf.train.GradientDescentOptimizer(0.001).minimize(cost)
tf.add_to_collection("train",train)

with tf.name_scope("accuracy"):
    correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init=tf.global_variables_initializer()
saver=tf.train.Saver(max_to_keep=3)


with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter("logs", sess.graph)
    writer.close()
    for i in range(300):
        batch_xs,batch_ys=mnist.train.next_batch(100)
        batch_xs=batch_xs.reshape([100,28,28])
        sess.run(train,feed_dict={x:batch_xs,y:batch_ys})
        if i%50==0:
            print(sess.run(accuracy,feed_dict={x:batch_xs,y:batch_ys}))
            save_path = saver.save(sess, "my_net/save_net",global_step=i)
            print("Save to path:", save_path)
