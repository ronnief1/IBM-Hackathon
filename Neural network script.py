
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
get_ipython().magic('matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.layers import Dropout
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_regression
from numpy import array
import timeit


# In[100]:


config = tf.ConfigProto(device_count={"CPU": 8})
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))


# In[65]:


seed = 7
np.random.seed(seed)


# In[44]:


df = pd.read_csv("HTRU_2.csv", header= None)
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]
Xtrain = train.iloc[:, 0:8]
Ytrain = train.iloc[:, 8]
Xtest = test.iloc[:, 0:8]
Ytest = test.iloc[:, 8]
Xtrain = (Xtrain - Xtrain.mean())/Xtrain.std()
Xtest = (Xtest - Xtest.mean())/Xtest.std()


# In[83]:


def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=8, activation='relu'))
    model.add(Dense(1, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[59]:


estimator = KerasClassifier(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)


# In[60]:


kfold = KFold(n_splits=3, shuffle=True, random_state=seed)


# In[67]:


start_time = timeit.default_timer()
results = cross_val_score(estimator, Xtrain, Ytrain, cv=kfold, n_jobs=1)
elapsed = timeit.default_timer() - start_time


# In[70]:


print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# In[99]:


model = Sequential()
model.add(Dense(8, input_dim=8, activation='relu'))
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  


# In[101]:


start_time = timeit.default_timer()
model.fit(Xtrain, Ytrain, epochs = 100, batch_size=5)
elapsed = timeit.default_timer() - start_time


# In[106]:


scores = model.evaluate(Xtrain, Ytrain)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

