import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

import string
import warnings
def concat_df(train_data, test_data):
# Returns a concatenated df of training and test set
 return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)
def drop_features(df):
 return df.drop([ 'Name', 'Embarked','Cabin'], axis=1)
def divide_df(all_data):
    # Returns divided dfs of training and test set
    return all_data.loc[:890], all_data.loc[891:].drop(['Survived'], axis=1)
def display_missing(df):    
    for col in df.columns.tolist():          
        print('{} column missing values: {}'.format(col, df[col].isnull().sum()))
    print('\n')

# Code adapted from https://www.kaggle.com/jeffd23/scikit-learn-ml-from-start-to-finish
def simplify_ages(df):
    
    bins = ( 0, 5, 12, 18, 25, 35, 60, 120)
  #  group_names = [ 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins )
    df.Age = categories
    return df
def simplify_fares(df):
   
    bins = (-0.1,0, 8, 15, 31, 1000)
  #  group_names = [ 'free','1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df.Fare, bins)
    df.Fare = categories
    return df
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_all = concat_df(df_train, df_test)
df_train.name = 'Training Set'
df_test.name = 'Test Set'
df_all.name = 'All Set'
dfs = [df_train, df_test]
# fill missing age
age_by_pclass_sex = df_all.groupby(['Sex', 'Pclass']).median()['Age']
# Filling the missing values in Age with the medians of Sex and Pclass groups
df_all['Age'] = df_all.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
# fill missing fare
med_fare = df_all.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
# Filling the missing value in Fare with the median Fare of 3rd class alone passenger
df_all['Fare'] = df_all['Fare'].fillna(med_fare) 
#fill missing and change cabin to desk
df_all['Deck'] = df_all['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
idx = df_all[df_all['Deck'] == 'T'].index
df_all.loc[idx, 'Deck'] = 'A'
df_all['Deck'] = df_all['Deck'].replace(['A', 'B', 'C'], 'ABC')
df_all['Deck'] = df_all['Deck'].replace(['D', 'E'], 'DE')
df_all['Deck'] = df_all['Deck'].replace(['F', 'G'], 'FG')
#############################################################new features based on relationship
df_all['Family_Size'] = df_all['SibSp'] + df_all['Parch'] + 1
df_all=simplify_ages(df_all)
df_all=simplify_fares(df_all)
df_all['Ticket_Frequency'] = df_all.groupby('Ticket')['Ticket'].transform('count')
df_all['Title'] = df_all['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
#df_all['Is_Married'] = 0
#df_all['Is_Married'].loc[df_all['Title'] == 'Mrs'] = 1
# group them
df_all['Title'] = df_all['Title'].replace(['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Miss/Mrs/Ms')
df_all['Title'] = df_all['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Dr/Military/Noble/Clergy')

#drop
df_all=drop_features(df_all)

df_train = df_all.loc[:890]
df_test = df_all.loc[891:]
dfs = [df_train, df_test]
#label encode
non_numeric_features = [ 'Age', 'Fare']

for df in dfs:
   for feature in non_numeric_features:        
        df[feature] = LabelEncoder().fit_transform(df[feature])
#for df in dfs:
    #display_missing(df)
#df_all['Deck'].value_counts()
df_all = concat_df(df_train, df_test)
print(df_all.head())
print(df_all['Title'].value_counts())
df_all.to_csv('data.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONE)