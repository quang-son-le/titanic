import numpy as np
import keras
sample_text = 'This is a sample sentence. is'
text=keras.preprocessing.text.one_hot(
    sample_text,20,split=' '
)
print(text)