

import keras

import pandas as pd
import csv
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

import numpy as np


# first neural network with keras tutorial
from numpy import loadtxt


def drop_features(df):
 return df.drop([ 'PassengerId', 'Sex','Ticket','Deck','Title','Survived'], axis=1)

df_all = pd.read_csv('data.csv')
passengerid=pd.DataFrame(df_all[891:],columns=['PassengerId'])
passengerid=passengerid.reset_index(drop=True)
passengerid.to_csv('passid.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONE)
#print(passengerid.shape)
df_all =drop_features(df_all)
df_train = df_all.loc[:890]
#print(df_train.describe())
df_test = df_all.loc[891:]

#print(df_test.describe())
#df_train.to_csv('datatrain.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONE)
df_test=df_test.drop([ 'Survived_1','Survived_2'], axis=1)
#df_test.to_csv('datatest.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONE)
#model
Y=pd.DataFrame(df_train,columns=['Survived_1','Survived_2'])
X=df_train.drop([ 'Survived_1','Survived_2'], axis=1)

#Y.to_csv('Y.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONE)
#X.to_csv('X.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONE)
# model
seed = 1234
np.random.seed(seed)
tf.set_random_seed(seed)

X=np.array(X.values, dtype='float32')
y=np.array(Y.values, dtype='float32')
test=np.array(df_test.values, dtype='float32')
indices = np.random.choice(len(X), len(X), replace=False)

X_values = X[indices]
y_values = y[indices]
# Creating a Train and a Test Dataset
test_size = 10
X_test = X_values[-test_size:]
X_train = X_values[:-test_size]
y_test = y_values[-test_size:]
y_train = y_values[:-test_size]
sess = tf.Session()

# Interval / Epochs
interval = 50
epoch = 400

# Initialize placeholders
X_data = tf.placeholder(shape=[None, 14], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 2], dtype=tf.float32)
# Output neurons : 3
hidden_layer_nodes = 10

# Create variables for Neural Network layers
w1 = tf.Variable(tf.random.normal(shape=[14,hidden_layer_nodes])) # Inputs -> Hidden Layer
b1 = tf.Variable(tf.random.normal(shape=[hidden_layer_nodes]))   # First Bias
w2 = tf.Variable(tf.random.normal(shape=[hidden_layer_nodes,2])) # Hidden layer -> Outputs
b2 = tf.Variable(tf.random.normal(shape=[2]))   # Second Bias
hidden_output = tf.nn.relu(tf.add(tf.matmul(X_data, w1), b1))
final_output = tf.nn.softmax(tf.add(tf.matmul(hidden_output, w2), b2))
# Cost Function
loss = tf.reduce_mean(-tf.reduce_sum(y_target * tf.math.log(final_output) , axis=0))#heeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeer

# Optimizer
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001, use_locking=False, name='GradientDescent').minimize(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False,name='Adam').minimize(loss)
# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Training
print('Training the model...')
for i in range(1, (epoch + 1)):
    sess.run(optimizer, feed_dict={X_data: X_train, y_target: y_train})
    if i % interval == 0:
       print('Epoch', i, '|', 'Loss:', sess.run(loss, feed_dict={X_data: X_train, y_target: y_train}))
#output=sess.run(final_output, feed_dict={X_data: test})
#print(np.array(output, dtype='uint32'))

# Prediction
print()

#output= np.rint(sess.run(final_output, feed_dict={X_data: test}))
#print(output[:,1])
#data = [passengerid["PassengerId"].astype(np.int), pd.DataFrame({'Survived':output[:,1].astype(np.int)})]

#result = pd.concat(data, axis=1)
#result.to_csv('network.csv', encoding='utf-8', index=False)
for i in range(len(X_test)):
  print('Actual:', y_test[i], 'Predicted:', np.rint(sess.run(final_output, feed_dict={X_data: [X_test[i]]})))
