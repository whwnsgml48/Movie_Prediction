import pandas as pd 
import tensorflow as tf 

train = pd.read_csv('Train.csv', encoding = 'cp949')

print(train.head())
print(train.columns)