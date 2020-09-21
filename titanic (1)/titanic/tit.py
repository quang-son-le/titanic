import pandas as pd
import numpy as np

from sklearn import preprocessing
def drop_features(df):
    return df.drop(['Ticket', 'Name', 'Embarked','Cabin'], axis=1)
def concat_df(train_data, test_data):
    # Returns a concatenated df of training and test set
    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)

def divide_df(all_data):
    # Returns divided dfs of training and test set
    return all_data.loc[:890], all_data.loc[891:].drop(['Survived'], axis=1)
def simplify_ages(df):
  
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df
def simplify_fares(df):
  
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df
def display_missing(df):    
    for col in df.columns.tolist():          
        print('{} column missing values: {}'.format(col, df[col].isnull().sum()))
    print('\n')
    
# Loading the dataset

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_all = concat_df(df_train, df_test)
dfs = [df_train, df_test]
df_train.name = 'Training Set'
df_test.name = 'Test Set'
df_all.name = 'All Set' 
#display_missing(df_all)
#fill age
group=df_all.groupby(['Sex', 'Pclass'])['Age']
med_fare = df_all.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
# Filling the missing value in Fare with the median Fare of 3rd class alone passenger
df_all['Fare'] = df_all['Fare'].fillna(med_fare)
df_all['Deck'] = df_all['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')

#idx = df_all[df_all['Deck'] == 'T'].index
#df_all.loc[idx, 'Deck'] = 'A'
df_all['Deck'] = df_all['Deck'].replace(['T'], 'A')


df_all['Deck'] = df_all['Deck'].replace(['A', 'B', 'C'], 'ABC')
df_all['Deck'] = df_all['Deck'].replace(['D', 'E'], 'DE')
df_all['Deck'] = df_all['Deck'].replace(['F', 'G'], 'FG')
# fill ages with median ages of sex and pclass group
age_by_pclass_sex = df_all.groupby(['Sex', 'Pclass']).median()['Age']
df_all['Age'] = df_all.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
df_all = simplify_ages(df_all)
df_all = simplify_fares(df_all)
df_all=drop_features(df_all)

#print('afer filling')
features = ['Fare',  'Age', 'Sex']
for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_all[feature])
        df_all[feature] = le.transform(df_all[feature])
display_missing(df_all)
df_all.to_csv('data.csv', encoding='utf-8', index=False)