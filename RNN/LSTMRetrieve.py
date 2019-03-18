import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

with tf.Session() as sess:
    ckpt=tf.train.latest_checkpoint("my_net")
    new_saver = tf.train.import_meta_graph(ckpt+".meta")
    new_saver.restore(sess, ckpt)

    graph = tf.get_default_graph()
    train = tf.get_collection('train')[0]
    predict = tf.get_collection("predict")[0]
    operation_name_list=[operation.name for operation in graph.as_graph_def().node]
    print(operation_name_list)
    print(tf.get_collection("inputx"))
    input_x=tf.get_collection("inputx")[0]
    print(graph.get_operations())
    #input_y=graph.get_tensor_by_name("input_y/input_y:0")
    input_y=graph.get_operation_by_name("input_y/input_y").outputs[0]

    writer = tf.summary.FileWriter("logs", sess.graph)
    writer.close()

    batch_xs, batch_ys = mnist.test.next_batch(100)
    batch_xs = batch_xs.reshape([100, 28, 28])

    res = sess.run(predict, feed_dict={input_x: batch_xs})

    correct_pred=tf.equal(tf.argmax(res,1),tf.argmax(batch_ys,1))
    accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

    saver = tf.train.Saver(max_to_keep=3)
    for i in range(300):
        batch_xs,batch_ys=mnist.train.next_batch(100)
        batch_xs=batch_xs.reshape([100,28,28])
        sess.run(train,feed_dict={input_x:batch_xs,input_y:batch_ys})
        if i%50==0:
            print(sess.run(accuracy,feed_dict={input_x:batch_xs,input_y:batch_ys}))
            save_path = saver.save(sess, "my_net/save_net",global_step=i)
            print("Save to path:", save_path)