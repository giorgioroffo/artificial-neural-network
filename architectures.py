from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

def buildANN(optimizer = 'adam', numUnits = 6):
    net = Sequential()
    net.add(Dense(units = numUnits, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    net.add(Dropout(rate=0.1))
    net.add(Dense(units = numUnits, kernel_initializer = 'uniform', activation = 'relu'))
    net.add(Dropout(rate=0.1))
    net.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    net.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return net
