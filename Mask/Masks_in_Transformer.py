import tensorflow as tf

padding_num = -1000
#keys 3*4 T_k*T_h  queries 2*4 T_q*T_h

k=[[[1,2,3,4],[3,4,5,6],[0,0,0,0]]]
q=[[[2,1,3,4],[0,0,0,0]]]

'''
inputs = tf.matmul(q, tf.transpose(k, [0,2, 1]))
print(inputs)
#keys mask 
masks = tf.sign(tf.reduce_sum(tf.abs(k), axis=-1)) #n*T_k
masks = tf.expand_dims(masks, 1) #n*1*T_k
masks = tf.tile(masks, [1,tf.shape(q)[1],1]) #n*T_q*T_k
paddings = tf.ones_like(inputs) * padding_num
outputs = tf.where(tf.equal(masks, 0), paddings, inputs)
print(outputs)

#sequence mask
diag_vals=diag_vals = tf.ones_like(inputs[0, :, :])
print(diag_vals)
tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0],1, 1])
paddings = tf.ones_like(masks) * padding_num
print(masks,paddings)
outputs = tf.where(tf.equal(masks, 0), paddings, inputs)
print(masks,outputs)
'''
#queries mask
softmax_inputs=tf.constant([[[0.3,0.7,0],[0.5,0.5,0]]],dtype=tf.float32)
masks = tf.sign(tf.reduce_sum(tf.abs(q), axis=-1)) #n*T_q
masks=tf.cast(masks,dtype=tf.float32)
masks = tf.expand_dims(masks, -1)  #n*T_q*1
masks = tf.tile(masks, [1, 1, tf.shape(k)[1]])  #n*T_q*T_k
# Apply masks to inputs
outputs = softmax_inputs * masks
print(outputs)
