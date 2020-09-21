import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Flatten
from keras import backend as K
from keras.models import Model
df_all = pd.read_csv('data.csv')
#print(df_all.head())
#df_all.Deck = pd.Categorical(df_all.Deck)
#df_all['code'] = df_all.Deck.cat.codes
#df_all.to_csv(r'data.csv')
embedding_size = 3
model = Sequential()
model.add(Embedding(input_dim = 4, output_dim = embedding_size, input_length = 1, name="embedding"))
model.add(Flatten())
model.add(Dense(50, activation="relu"))
model.add(Dense(15, activation="sigmoid"))
model.add(Dense(1))
model.compile(loss = "mse", optimizer = "adam", metrics=["accuracy"])
print(model.summary())
model.fit(x = df_all[['code']].to_numpy(), y=df_all[['Fare']].to_numpy() , epochs = 50, batch_size = 4,verbose=0)
print('result')
layer_name = 'embedding'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(np.array([0,1,2,3]))
print(intermediate_output )
