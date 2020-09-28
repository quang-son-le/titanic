neural networkk and deep embedded encoding to solve titanic survival prediction on kaggle
data pre processing (pre_procees.py) is adopted from 
https://www.kaggle.com/gunesevitan/titanic-advanced-feature-engineering-tutorial?fbclid=IwAR0TgrZslgDKmHP5n7yQyzmArNhOuOSJcrgUADrccfk-lJkdqbwRvCMlpwo

in order to run pre_process.py to create data1.csv

there are 3 nomimal data groups here are Deck, Title and Sex. Because Sex intuitively doesn't have any relationship with any others, so only 2 are encoded. Sex is one hot encodded
Run Title_encoding.py, get data2.csv

Run Deck_encoding.py get data.csv

Run keras_t.py to predict, output is network_keras.csv

other classifiers (run with data.csv) is adopted from 
https://github.com/gtraskas/titanic_prediction/blob/master/titanic_prediction.ipynb?fbclid=IwAR1zd1Y0LsKFM68ir724Kkv2nkiRoRwDwkVf8IwIaO-5PM65pl4HjloXsHk

to compare

score is about 0.77. Tuning network and apply Pca (will throw away some features of deep encoding) wont improve much socre

any feedback is welcomed, email: officialquangsonle@gmail.com
