



import pandas as pd
import csv



from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
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
X=np.array(X.values, dtype='float32')
y=np.array(Y.values, dtype='float32')
test=np.array(df_test.values, dtype='float32')
# model
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.01)
model = Sequential()

model.add(Dense(14, input_shape=(14,), activation='relu', name='fc1'))
model.add(Dense(40, activation='sigmoid', name='fc2'))#40
model.add(Dense(2, activation='softmax', name='output'))

# Adam optimizer with learning rate of 0.001
optimizer = Adam(lr=0.001)
model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

print('Neural Network Model Summary: ')
print(model.summary())

# Train the model
model.fit(train_x, train_y, verbose=2, batch_size=5, epochs=150)#150

# Test on unseen data

results = model.evaluate(test_x, test_y)
output=np.rint(model.predict_classes(test))
#print(output.shape)
#print(output)
#print(output.shape)
data = [passengerid["PassengerId"].astype(np.int), pd.DataFrame({'Survived':output.astype(np.int)})]
result = pd.concat(data, axis=1)
result.to_csv('network_keras.csv', encoding='utf-8', index=False)
print('Final test set loss: {:4f}'.format(results[0]))
print('Final test set accuracy: {:4f}'.format(results[1]))