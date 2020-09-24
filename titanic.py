def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import os
import sys
stderr=sys.stderr
sys.stderr=open(os.devnull, 'w')
import keras
sys.stderr=stderr
import pandas as pd
import csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from time import time

def drop_features(df):
 return df.drop([ 'PassengerId', 'Sex','Ticket','Deck','Title','Survived'], axis=1)

df_all = pd.read_csv('data.csv')
passengerid=pd.DataFrame(df_all[891:],columns=['PassengerId'])
passengerid=passengerid.reset_index(drop=True)
passengerid.to_csv('passid.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONE)
print(passengerid.shape)
df_all =drop_features(df_all)
df_train = df_all.loc[:890]
#print(df_train.describe())
df_test = df_all.loc[891:]

#print(df_test.describe())
#df_train.to_csv('datatrain.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONE)
df_test=df_test.drop([ 'Survived_1','Survived_2'], axis=1)
#df_test.to_csv('datatest.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONE)
#model
Y=pd.DataFrame(df_train,columns=['Survived_2'])
X=df_train.drop([ 'Survived_1','Survived_2'], axis=1)
#Y.to_csv('Y.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONE)
#X.to_csv('X.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONE)
# model
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.2,
                                                    random_state=23)

# Check the split printing the shape of each set.
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
nb = GaussianNB()
svc = SVC()
dtc = DecisionTreeClassifier()
knc = KNeighborsClassifier()
rfc = RandomForestClassifier()
abc = AdaBoostClassifier()

# Create a dictionary of classifiers to choose from.
classifiers = {"GaussianNB": nb, "SVM": svc, "Decision Trees": dtc, 
               "KNN": knc, "Random Forest": rfc, "AdaBoost": abc}
def test_clfs(clf,name):
    
    # Create the KFold cross validation iterator.
    kf = KFold(n_splits=50, shuffle=True, random_state=23)
    
    outcomes = []
    fold = 0
    for train_index, test_index in kf.split(X):
        t0 = time()
        fold += 1
        X_train, X_test = X.values[train_index], X.values[test_index]
        y_train, y_test = Y.values[train_index], Y.values[test_index]
           
        # Fit the classifier to the data.
        
            
        #outcomes.append(accuracy)
            
    mean_outcome = np.mean(outcomes)
    clf.fit(X_train, y_train)
            
        # Create a set of predictions.
    predictions = clf.predict(df_test)
    
    #predictions=predictions.reshape(-1,1)
    data = [passengerid["PassengerId"], pd.DataFrame({'Survived':predictions})]
    #headers = ["PassengerId", "Survivied"]
    print(passengerid.shape)
    print(predictions.shape)
    result = pd.concat(data, axis=1)
    
    #predictions=pd.DataFrame({'PassengerId': passengerid[891:], 'Survivied':predictions},index=[0])
    result.to_csv('{}.csv'.format(name), encoding='utf-8', index=False)
        # Evaluate predictions with accuracy score.
    accuracy = clf.score(X_test, y_test) 
    print("\nMean Accuracy: {0}".format(accuracy))    
    # Print the results.
    #print("\nMean Accuracy: {0}".format(mean_outcome))
    #print("\nTime passed: ", round(time() - t0, 3), "s\n")

for name, clf in classifiers.items():
    print("#"*55)
    print(name)
    test_clfs(clf,name)
    