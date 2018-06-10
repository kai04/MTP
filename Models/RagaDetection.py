import pandas as pd
import json
import math
import pickle as p
import tensorflow as tf
import numpy as np
import utils
import datetime 
from tensorflow import summary as summ
# set variables
tweet_size = 7284
hidden_size = 100
vocab_size = 12
batch_size = 64
num_classes=14

# this just makes sure that all our following operations will be placed in the right graph.
tf.reset_default_graph()

# create a session variable that we can run later.
session = tf.Session()

# make placeholders for data we'll feed in
tweets = tf.placeholder(tf.float32, [None, tweet_size, vocab_size])
labels = tf.placeholder(tf.float32, [None,num_classes])

# make the lstm cells, and wrap them in MultiRNNCell for multiple layers
lstm_cell_1 = tf.contrib.rnn.LSTMCell(hidden_size)
lstm_cell_2 = tf.contrib.rnn.LSTMCell(hidden_size)
multi_lstm_cells = tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell_1, lstm_cell_2] , state_is_tuple=True)

# define the op that runs the LSTM, across time, on the data
_, final_state = tf.nn.dynamic_rnn(multi_lstm_cells, tweets, dtype=tf.float32)


def second_NN(input_, output_size,hidden_nodes, name, init_bias=0.0):
    shape = input_.get_shape().as_list()

    # Variables for two group of weights between the three layers of the network
    with tf.variable_scope(name):
        W1 = tf.get_variable("weight_matrix1", [shape[-1], hidden_nodes], tf.float32, tf.random_normal_initializer(stddev=1.0 / math.sqrt(shape[-1])))
        if init_bias is None:
            A1 = tf.sigmoid(tf.matmul(input_, W1))
        else:
            b1 = tf.get_variable("bias", [hidden_nodes], initializer=tf.constant_initializer(init_bias))
            A1 = tf.sigmoid(tf.matmul(input_, W1)+b1)
            
        W2 = tf.get_variable("weight_matrix2", [hidden_nodes, output_size], tf.float32, tf.random_normal_initializer(stddev=1.0 / math.sqrt(shape[-1])))
        return tf.matmul(A1,W2)

# define that our final sentiment logit is a linear function of the final state 
# of the LSTM
logits = second_NN(final_state[-1][-1],num_classes,120,name="output")
prediction = tf.nn.softmax(logits)
#print("type of prediction:",type(prediction))
#print("shape of prediction:",tf.shape(prediction))
#print("logits Shape:",np.shape(logits))
#print("prediction Shape:",np.shape(prediction))
#print("labels Shape:",np.shape(labels))
# define cross entropy loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))

# round our actual probabilities to compute error
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# define our optimizer to minimize the loss
optimizer = tf.train.AdamOptimizer().minimize(loss)

# initialize any variables
tf.global_variables_initializer().run(session=session)

# load our data and separate it into tweets and labels

train_data_df=pd.read_csv("./Data/traindata_temp.csv",sep=",")
train_labels_df=train_data_df[train_data_df.columns[0:1]]
train_data_df.drop(train_data_df.columns[[0]], axis=1)
#print("train data:",train_data_df.iloc[0])
#print("train label:",train_labels_df)
train_data=train_data_df.values
train_labels_temp=train_labels_df.values
train_labels_temp1=np.array([int(t[0]) for t in train_labels_temp])
#print("train_labels_temp:",train_labels_temp[0][0])
#print("train_labels_temp:",np.shape(train_labels_temp1))
#train_labels_df.to_csv("./Data/check.csv", encoding='utf-8', index=False)
train_tweets=train_data_df.values
train_labels = (tf.one_hot(train_labels_temp1, num_classes)).eval(session=session)

#print("train_tweets shape:",np.shape(train_tweets))
#print("train_labels shape:",np.shape(train_labels))

test_data_df=pd.read_csv("./Data/testdata_temp.csv",sep=",")
test_data=test_data_df.values
test_labels_df = test_data_df[test_data_df.columns[0:1]]
test_data_df.drop(test_data_df.columns[[0]], axis=1)


#test_data_df.drop(test_data_df.columns[[0]],axis=1)
test_tweets=test_data_df.values

one_hot_test_tweets = (tf.one_hot(test_tweets, vocab_size)).eval(session=session)
test_labels=test_labels_df.values
#print("testdata shape:",np.shape(test_tweets))
#print("testlabels shape:",np.shape(test_labels))


#train_data = json.load(open('data/trainTweets_preprocessed.json', 'r'))
#train_data = list(map(lambda row:(np.array(row[0],dtype=np.int32),str(row[1])),train_data))
#
#train_tweets = np.array([t[0] for t in train_data])
#train_labels_temp = np.array([int(t[1]) for t in train_data])
#train_labels = (tf.one_hot(train_labels_temp, num_classes)).eval(session=session)
#
#test_data = json.load(open('data/testTweets_preprocessed.json', 'r'))
#test_data = map(lambda row:(np.array(row[0],dtype=np.int32),str(row[1])),test_data)
## we are just taking the first 1000 things from the test set for faster evaluation
#test_data = test_data[0:1000] 
#test_tweets = np.array([t[0] for t in test_data])
#one_hot_test_tweets = (tf.one_hot(test_tweets, vocab_size)).eval(session=session)
#test_labels = np.array([int(t[1]) for t in test_data])

tf.summary.scalar("training_loss",loss)
tf.summary.scalar("train_accuracy",accuracy)
tf.summary.histogram("state",final_state)   

summaryMerged=tf.summary.merge_all()
filename="./summary_log/run"+(datetime.datetime.now()).strftime("%Y-%m-%d--%H-%M-%s")
writer = tf.summary.FileWriter(filename, session.graph )

# we'll train with batches of size 128.  This means that we run 
# our model on 128 examples and then do gradient descent based on the loss
# over those 128 examples.
num_steps = 50

for step in range(num_steps):
    # get data for a batch
#    print("step:",step)
    offset = (step * batch_size) % (len(train_data) - batch_size)
    batch_tweets = (tf.one_hot(train_tweets[offset : (offset + batch_size)], vocab_size)).eval(session=session)
    batch_labels = train_labels[offset : (offset + batch_size)]
#    print("batch_tweets shape:",np.shape(batch_tweets))
#    print("batch_labels shape:",np.shape(batch_labels))    
    # put this data into a dictionary that we feed in when we run 
    # the graph.  this data fills in the placeholders we made in the graph.
    data = {tweets: batch_tweets, labels: batch_labels}
    
    # run the 'optimizer', 'loss', and 'pred_err' operations in the graph
    _, loss_value_train, error_value_train,state,sumOut,pred_val,true_val = session.run(
      [optimizer, loss, accuracy,final_state,summaryMerged,tf.argmax(prediction,1),tf.argmax(labels,1)], feed_dict=data)
      
    
    # print stuff every 50 steps to see how we are doing
#    print("pred_val type: ",type(pred_val))    
#    print("true_val type: ",type(true_val))
    if (step % 50 == 0):
#        print("step:",step)
        print("Minibatch train loss at step", step, ":", loss_value_train)
        print("Minibatch train accuracy: %.3f%%" % error_value_train)
#        print("true value:",true_val)
#        print("predicted value:",pred_val)
        pred_val1=pred_val[np.newaxis]
        pred_val1=pred_val1.T
        true_val1=true_val[np.newaxis]
        true_val1=true_val1.T        
        df1=pd.DataFrame(pred_val1)
        df2=pd.DataFrame(true_val1)
        df1.to_csv("./Data/pred_val.csv", encoding='utf-8', index=False)
        df2.to_csv("./Data/true_val.csv", encoding='utf-8', index=False)
        # get test evaluation
        test_loss = []
        test_error = []
        for batch_num in range(int(len(test_data)/batch_size)):
            test_offset = (batch_num * batch_size) % (len(test_data) - batch_size)
            test_batch_tweets = one_hot_test_tweets[test_offset : (test_offset + batch_size)]
            test_batch_labels_temp = test_labels[test_offset : (test_offset + batch_size)]
            test_batch_labels_temp1=np.array([int(t[0]) for t in test_batch_labels_temp])
            test_batch_labels = (tf.one_hot(test_batch_labels_temp1, num_classes)).eval(session=session)
            data_testing = {tweets: test_batch_tweets, labels: test_batch_labels}
            loss_value_test, error_value_test = session.run([loss, accuracy], feed_dict=data_testing)
            test_loss.append(loss_value_test)
            test_error.append(error_value_test)
        
    writer.add_summary(sumOut,step)
