from __future__ import print_function
import tensorflow as tf
import numpy as np

#gpu_memory = tf.GPUOptions(per_process_memory_fraction=0.25)
#config=tf.ConfigProto(gpu_options=gpu_memory)

def get_layer_shape(layer):
	thisshape = tf.Tensor.get_shape(layer)
	ts = [thisshape[i].value for i in range(len(thisshape))]
	return ts

def BasicLSTM(varscope,nodes,rnn_input,input_states=None):
	with tf.variable_scope(varscope):
		cell = tf.nn.rnn_cell.BasicLSTMCell(nodes,state_is_tuple=True)
		if(input_states is None):
			lstm_out, lstm_state = tf.nn.dynamic_rnn(cell, rnn_input, dtype=tf.float32)
		else:
			lstm_out, lstm_state = tf.nn.dynamic_rnn(cell, rnn_input, initial_state=input_states)
	return lstm_out,lstm_state

def LSTM(varscope,nodes,rnn_input,input_states=None):
	with tf.variable_scope(varscope):
		cell = tf.nn.rnn_cell.LSTMCell(nodes,state_is_tuple=True)
		if(input_states is None):
			lstm_out, lstm_state = tf.nn.dynamic_rnn(cell, rnn_input, dtype=tf.float32)
		else:
			lstm_out, lstm_state = tf.nn.dynamic_rnn(cell, rnn_input, initial_state=input_states)
	return lstm_out,lstm_state

def BLSTM(nb_nodes,input,name,seq_len=None,return_state=False):
    cell_name=name+"_def"
    nb_nodes=int(nb_nodes/2)
    with tf.variable_scope(cell_name):
        f_cell = tf.nn.rnn_cell.LSTMCell(nb_nodes, state_is_tuple=True)
        b_cell = tf.nn.rnn_cell.LSTMCell(nb_nodes, state_is_tuple=True)
    op_name=name+"_op"
    with tf.variable_scope(op_name):
        if(seq_len is not None):
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(f_cell, b_cell, input, sequence_length=seq_len, dtype=tf.float32)
        else:
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(f_cell, b_cell, input, dtype=tf.float32)

    merge=tf.concat(outputs,2)
    if(return_state):
        return merge,output_states
    else:
        return merge

def Convolution2D(input,filter_dim, nbfilters,stride,layername):
    #filter_dim [w,h] stride [w,h]
    filter_w=filter_dim[0]
    filter_h=filter_dim[1]
    shape=get_layer_shape(input)
    filter_in=shape[-1]
    #print(filter_w,filter_h,filter_in,nbfilters)
    filter=tf.Variable(tf.truncated_normal([filter_h,filter_w,filter_in,nbfilters],name=layername+"_Filter"))
    shift=[1,stride[1],stride[0],1]
    convolved = tf.nn.conv2d(input,filter,shift,padding='SAME',name=layername+"_convolution2d")
    convolved=tf.nn.relu(convolved)
    return convolved

def Convolution1D(input,filter_width,nbfilters,stride,layername,activation=True):
    shape = get_layer_shape(input)
    filter_in = shape[-1]
    filter = tf.Variable(tf.truncated_normal([filter_width, filter_in, nbfilters], name=layername + "_Filter"))
    #shift = [1, stride, 1]
    convolved = tf.nn.conv1d(input, filter, stride, padding='SAME', name=layername+"_convolution1d")
    if(activation):
        convolved = tf.nn.relu(convolved)
    return convolved

def Pooling1D(input,poolsize,stride,layername):
    input_4d=tf.expand_dims(input,axis=1)
    ksize=[1,1,poolsize,1]
    shift=[1,1,stride,1]
    pooled=tf.nn.max_pool(input_4d,ksize,shift,padding='SAME',name=layername+"_maxpool")
    input_3d=tf.squeeze(pooled,axis=1)
    return input_3d

def Pooling2D(input,poolsize,stride,layername):
    ksize=[1,poolsize[1],poolsize[0],1]
    shift=[1,stride[1],stride[0],1]
    pooled=tf.nn.max_pool(input,ksize,shift,padding='SAME',name=layername+"_maxpool")
    return pooled

def FullyConnected(input,nbnodes,layername):
    shape=get_layer_shape(input)
    in_dim=shape[-1]
    #print("In dimesion ",in_dim)
    W=tf.Variable(tf.truncated_normal([in_dim,nbnodes]),name=layername+"_W")
    B=tf.constant(0.1,shape=[nbnodes],name=layername+"_B")
    dense_out=tf.matmul(input,W)+B
    dense_confidence=tf.nn.softmax(dense_out)
    return dense_out,dense_confidence

