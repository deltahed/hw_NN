import time
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, scale

plt.rcParams['figure.figsize']=(15,10)

t0 = time.time()

#Data preprocessing
df = pd.read_excel(\
"C:/Users/Jongho/Desktop/KAIST/2017_3Q/FE568_빅데이터를이용한신용위험분석_한철우/Assignment/data/train.xlsx")

data = df

data.columns = list(data.iloc[0,:])
data = data[1:]
dummyCol = data.columns[[1,2,3]]
data[dummyCol] = data[dummyCol].astype(str)

for i in dummyCol:
    tmp = pd.get_dummies(data[i])
    tmp.columns = i + tmp.columns
    del data[i]
    data = pd.concat([data, tmp], axis=1)

sns.kdeplot(data[data['default'] == 0]['LIMIT_BAL'], label='Good')
sns.kdeplot(data[data['default'] == 1]['LIMIT_BAL'], label='Bad')
plt.legend()
plt.show()

graph = sns.jointplot(x='LIMIT_BAL', y='AGE', data=data[data['default'] == 0], kind='kde', color='red')

graph.x = data[data['default'] == 1]['LIMIT_BAL']
graph.y = data[data['default'] == 1]['AGE']
graph.plot_joint(plt.scatter, marker='.', c='b', s=5)
plt.show()

##Divide trainning & validation sets
#train = data[:12000]
#validation = data[12000:]
#
#xTrain = train.loc[:,train.columns != 'default'].values
#xValidation = validation.loc[:,validation.columns != 'default'].values
#    
#yTrain = np.array(train['default'].values, dtype=np.int64)
#yValidation = np.array(validation['default'].values, dtype=np.int64)
#
##Data normalarization
#xTrain = scale(xTrain)
#xValidation = scale(xValidation)
#
##Building graphs
#x = tf.placeholder(tf.float32, [None, xTrain.shape[1]], name='X')
#y = tf.placeholder(tf.float32, [None], name='y')
#
#Layer1_node, Layer1_keep = (50, 0.8)
#Layer2_node, Layer2_keep = (15, 0.8)
#Layer3_node, Layer3_keep = (20, 0.8)
#
#with tf.name_scope('Layer1') as scope:
#    W1 = tf.Variable(tf.truncated_normal([xTrain.shape[1], Layer1_node], stddev=0.1), name='W1')
#    b1 = tf.Variable(tf.zeros([Layer1_node]), name='b1')
#    h1 = tf.nn.relu(tf.matmul(x, W1) + b1)
##    h1_drop = h1
#    h1_drop = tf.nn.dropout(h1, keep_prob=Layer1_keep)
#
#with tf.name_scope('Layer2') as scope:
#    W2 = tf.Variable(tf.truncated_normal([Layer1_node, Layer2_node], stddev=0.1), name='W2')
#    b2 = tf.Variable(tf.zeros([Layer2_node]), name='b2')
#    h2 = tf.nn.relu(tf.matmul(h1_drop, W2) + b2)
##    h2_drop = h2
#    h2_drop = tf.nn.dropout(h2, keep_prob=Layer2_keep)
#    
#with tf.name_scope('Layer3') as scope:
#    W3 = tf.Variable(tf.truncated_normal([Layer2_node, Layer3_node], stddev=0.1), name='W3')
#    b3 = tf.Variable(tf.zeros([Layer3_node]), name='b3')
#    h3 = tf.nn.relu(tf.matmul(h2_drop, W3) + b3)
##    h3_drop = h3
#    h3_drop = tf.nn.dropout(h3, keep_prob=Layer3_keep)
#    
#with tf.name_scope('Layer4') as scope: 
#    W4 = tf.Variable(tf.truncated_normal([Layer3_node, 1], stddev=0.1), name='W4')
#    b4 = tf.Variable(tf.zeros([1]), name='b4')
#    y_hat = tf.sigmoid(tf.to_double(tf.reduce_sum(tf.matmul(h3_drop, W4), [1]) + b4))
#    
#    
#with tf.name_scope('loss') as scope:
#    cost = -tf.to_double(y)*tf.log(y_hat) - (1.0 - tf.to_double(y))*tf.log(1.0 - y_hat)
#    #L1reg = 1 * tf.reduce_sum(tf.abs(W4))
#    J = tf.reduce_mean(tf.to_float(cost))
#
#with tf.name_scope('train') as scope:
#    train_step = tf.train.GradientDescentOptimizer(0.3).minimize(J)
#
##Machine Learning
#sess = tf.Session()
#sess.run(tf.global_variables_initializer())
#
#iteration = 100
#batch_size = 1000
#loss_log = []
#for i in range(iteration):
#    for j in range(0, len(xTrain), batch_size):
#        loss_value, _ = sess.run([J, train_step], feed_dict={x: xTrain[j:j+batch_size], y:yTrain[j:j+batch_size]})
#        #loss_log.append(sess.run(J, feed_dict={x: xValidation, y:yValidation}))
#        loss_log.append(loss_value)
#        print(loss_value)
#
#plt.plot(loss_log)
#plt.xlabel('iteration')
#plt.ylabel('ylabel')
#plt.show()
#
#t1 = time.time()
#print("Computation Time : ", t1-t0)
#
#h11 = tf.nn.relu(tf.matmul(x, W1) + b1)
#h22 = tf.nn.relu(tf.matmul(h11, W2) + b2)
#h33 = tf.nn.relu(tf.matmul(h22, W3) + b3)
#y_hat2 = tf.sigmoid(tf.matmul(h33, W4) + b4)
#prediction = tf.greater_equal(y_hat2, 0.5)
#pred = sess.run(prediction, feed_dict={x:xValidation})
#print("Predicted default is True : ", pred.sum())
#
##is_correct = tf.equal(pred, yValidation)
##accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
##
##print("Accuracy : ", sess.run(accuracy, feed_dict={x:xValidation, y:yValidation}))