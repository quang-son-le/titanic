#neural network and deep embedded encoding to solve titanic survival prediction on Kaggle

The data preprocessing (pre_process.py) is adapted from a [Kaggle tutorial](https://www.kaggle.com/gunesevitan/titanic-advanced-feature-engineering-tutorial?fbclid=IwAR0TgrZslgDKmHP5n7yQyzmArNhOuOSJcrgUADrccfk-lJkdqbwRvCMlpwo) on advanced feature engineering for the Titanic dataset. To execute pre_process.py and create data1.csv, note that there are three nominal data groups: Deck, Title, and Sex. Since Sex is intuitively unrelated to the others, only Deck and Title are encoded. Sex is one-hot encoded.

Execute Title_encoding.py to get data2.csv.

Execute Deck_encoding.py to get data.csv.

Execute keras_t.py to predict, output is network_keras.csv

other classifiers (run with data.csv) are adopted from  [titanic prediction](https://github.com/gtraskas/titanic_prediction/blob/master/titanic_prediction.ipynb?fbclid=IwAR1zd1Y0LsKFM68ir724Kkv2nkiRoRwDwkVf8IwIaO-5PM65pl4HjloXsHk)

to compare:

the score is about 0.77. Tuning the network and applying Pca (which will throw away some features of deep encoding) won't improve the score

any feedback is welcomed, email: officialquangsonle@gmail.com
