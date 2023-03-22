import numpy as np
import tensorflow as tf
from keras.datasets import fashion_mnist
from sklearn import linear_model

(Xtrain,ytrain),(Xtest,ytest) = fashion_mnist.load_data()

ntrain = Xtrain.shape[0]
ntest = Xtest.shape[0]
nrow = Xtrain.shape[1]
ncol = Xtrain.shape[2]


npixels = nrow*ncol
Xtrain = 2*(Xtrain/255 - 0.5)
Xtrain = Xtrain.reshape((ntrain,npixels))

Xtest = 2*(Xtest/255 - 0.5)
Xtest = Xtest.reshape((ntest,npixels))


model = linear_model.LogisticRegression(verbose=10, solver='sag', max_iter=250)
model.fit(Xtrain,ytrain)


prediction = model.predict(Xtest)
successrate = np.mean(prediction == ytest)
print('Accuaracy: {0:f}'.format(successrate))
