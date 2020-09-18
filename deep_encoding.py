import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Flatten
df_all = pd.read_csv('data.csv')
print(df_all.head())
df_all.Deck = pd.Categorical(df_all.Deck)
df_all['code'] = df_all.Deck.cat.codes

embedding_size = 3
model = Sequential()
model.add(Embedding(input_dim = 4, output_dim = 3, input_length = 1, name="embedding"))
model.add(Flatten())
model.add(Dense(50, activation="relu"))
model.add(Dense(15, activation="relu"))
model.add(Dense(1))
model.compile(loss = "mse", optimizer = "adam", metrics=["accuracy"])
model.fit(x = df_all[['code']].to_numpy(), y=df_all[['Fare']].to_numpy() , epochs = 50, batch_size = 4)
