import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Flatten
from keras import backend as K
from keras.models import Model
import csv
df_all = pd.read_csv('data1.csv')
#print(df_all.head())
#df_all.Deck = pd.Categorical(df_all.Deck)
#df_all['code'] = df_all.Deck.cat.codes

#model for title 
print('now train for title')
embedding_size = 3
model = Sequential()
model.add(Embedding(input_dim = 4, output_dim = embedding_size, input_length = 1, name="embedding"))
model.add(Flatten())
model.add(Dense(50, activation="relu"))
model.add(Dense(15, activation="sigmoid"))
model.add(Dense(5))
model.compile(loss = "mse", optimizer = "adam", metrics=["accuracy"])
print(model.summary())
model.fit(x = df_all[['Title']].to_numpy(), y=df_all[['Pclass','Sex_1','Sex_2','Age','Fare']].to_numpy() , epochs = 60, batch_size = 4,verbose=0)
model.save('title_model')
model = keras.models.load_model('title_model')

#result
print('result for title')
layer_name = 'embedding'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(df_all[['Title']]).reshape(-1,3)
print(intermediate_output)
n = 3
cols = ['{}_{}'.format('Title', n) for n in range(1, n + 1)]
print(cols)
encoded_df = pd.DataFrame(intermediate_output, columns=cols)
df_all = pd.concat([df_all, encoded_df], axis=1) 
#print(intermediate_output )
#model for Deck


df_all.to_csv('data2.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONE)