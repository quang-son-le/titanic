import numpy as np
import pandas as pd
def concat_df(train_data, test_data):
# Returns a concatenated df of training and test set
 return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)
def drop_features(df):
 return df.drop(['Ticket', 'Name', 'Embarked','Cabin'], axis=1)
def divide_df(all_data):
    # Returns divided dfs of training and test set
    return all_data.loc[:890], all_data.loc[891:].drop(['Survived'], axis=1)
def display_missing(df):    
    for col in df.columns.tolist():          
        print('{} column missing values: {}'.format(col, df[col].isnull().sum()))
    print('\n')
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
df_all=drop_features(df_all)

df_train, df_test = divide_df(df_all)
dfs = [df_train, df_test]

#for df in dfs:
    #display_missing(df)
#df_all['Deck'].value_counts()
print(df_all.head())
print(df_all['Deck'].value_counts())
#df_all.to_csv(r'data.csv')