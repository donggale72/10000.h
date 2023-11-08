from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def housing_loader(cols=[], norm=True):

  df = pd.read_csv('https://raw.githubusercontent.com/rasbt/'
                 'python-machine-learning-book-3rd-edition/'
                 'master/ch10/housing.data.txt',
                 header=None,
                 sep='\s+')

  df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 
                'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
                'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
  print(df.shape)

  if cols:
    X = df[cols].values
  else:
    X = df.drop(['MEDV'], axis=1)
  y = df['MEDV'].values

  X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1)

  if norm:
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    sc_x.fit(X_train)
    sc_y.fit( np.expand_dims(y_train, 1) )

    X_train = sc_x.fit_transform(X_train)
    y_train = sc_y.fit_transform( np.expand_dims(y_train, 1) ).flatten()
    X_test = sc_x.fit_transform(X_test)
    y_test = sc_y.fit_transform( np.expand_dims(y_test, 1) ).flatten()
  return X_train, X_test, y_train, y_test