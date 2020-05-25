# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix
from sklearn.compose import ColumnTransformer 

import keras
from keras.wrappers.scikit_learn import KerasClassifier


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

def buildANN(optimizer = 'adam', numUnits = 6):
    model = Sequential()
    model.add(Dense(units = numUnits, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    model.add(Dropout(rate=0.1))
    model.add(Dense(units = numUnits, kernel_initializer = 'uniform', activation = 'relu'))
    model.add(Dropout(rate=0.1))
    model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model


# Importing the dataset
dataset = pd.read_csv('dataset.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
labelEncoder = LabelEncoder()
X[:, 1] = labelEncoder.fit_transform(X[:, 1])
labelEncoder = LabelEncoder()
X[:, 2] = labelEncoder.fit_transform(X[:, 2])

transformer = ColumnTransformer(
    transformers=[
        ("Country",         
         OneHotEncoder(), 
         [1] 
         )
    ], remainder='passthrough'
)

X = transformer.fit_transform(X)

# Avoid dummy variable trap on idx 
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Create keras wrapper fo the ANN
annModel = KerasClassifier(build_fn = buildANN)

# Parameter Tuning
parameters = {'batch_size': [25, 32],
              'epochs': [50, 100, 500],
              'optimizer': ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator = annModel,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_

best_accuracy = grid_search.best_score_

print('best_accuracy= ' + str(best_accuracy))


print('Fitting the ANN to the Training set with the parameters found by GridSearchCV')

# Fitting the ANN to the Training set
classifier = buildANN(optimizer=best_parameters.get('optimizer'))
classifier.fit(X_train, y_train, best_parameters.get('batch_size'), epochs = best_parameters.get('epochs'))


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

print('Confusion matrix: ')

cm = confusion_matrix(y_test, y_pred)

print(cm)

acc =  100 * (cm[0,0]+cm[1,1]) / np.sum(cm)

print('Test accuracy: %.2f%% ' % (acc))

