#!/usr/bin/env python
# coding: utf-8

# In[14]:


import os
import warnings
import keras


from time import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# Import the libs for test split,KNN, svm and DecisionTree Classifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

# Import the libs for Tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow import feature_column
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import utils 
from yellowbrick.model_selection import LearningCurve


from ucimlrepo import fetch_ucirepo
from IPython.display import clear_output

matplotlib.use('agg')
get_ipython().run_line_magic('matplotlib', 'inline')

warnings.filterwarnings("ignore", category=UserWarning)


# In[15]:


dry_bean = fetch_ucirepo(id=602)
# data (as pandas dataframes) 
X = dry_bean.data.features 
y = dry_bean.data.targets 
# metadata 
print(dry_bean.metadata) 

# variable information 
print(dry_bean.variables)


# In[16]:


df = pd.concat([X, y], axis=1)
df.head()


# In[17]:


feature_columns = df.columns.to_numpy()[:-1]
X = df[feature_columns]
y = df['Class']


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.2, random_state=142)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.2, random_state=42)


# In[19]:


sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_train = pd.DataFrame(X_train_scaled)

X_val_scaled = sc.fit_transform(X_val)
X_val = pd.DataFrame(X_val_scaled)

X_test_scaled = sc.fit_transform(X_test)
X_test = pd.DataFrame(X_test_scaled)



# In[20]:


# Label Encode the data
lbl = LabelEncoder()
y_train_encoded = lbl.fit_transform(y_train)
y_val_encoded = lbl.fit_transform(y_val)
y_testencoded = lbl.fit_transform(y_test)
# one hot encode
y_train = to_categorical(y_train_encoded)
y_val = to_categorical(y_val_encoded)
y_test= to_categorical(y_test_encoded)


# In[21]:


fractions = np.arange(.2,.8,.05)
n_samples = len(X_train)
subsets = []

for frac in fractions:
    size = int(frac * n_samples)
    indices = np.random.choice(len(X_train), size, replace=False)    
    X_subset = X_train.iloc[indices]
    '''
    print(indices)
    print(X_train.index)
    print(len(X_train))
    '''
    y_subset = y_train[indices]
    
    subsets.append((X_subset, y_subset))


# In[18]:


'''
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_train = pd.DataFrame(X_train_scaled, index=X_train.index)

X_train_partial = []
y_train_partial = []
for subset in subsets:
    X_train_partial_samples, y_train_partial_samples = subset
    X_train_partial_samples_scaled = sc.fit_transform(X_train_partial_samples)
    X_train_partial.append(pd.DataFrame(X_train_partial_samples_scaled))
'''


# In[19]:


'''
x_ss = []
sc = StandardScaler()
for x in X_train_sample:
    x_ss.append(pd.DataFrame(sc.fit_transform(x),index=x.index))
'''

    


# In[20]:


y_val


# In[22]:


# Create the output dir
output_dir = '/tmp/logs'
os.makedirs(output_dir, exist_ok=True)
os.path.exists(output_dir)


# In[24]:


class PlotLearning(keras.callbacks.Callback):
    """
    Callback to plot the learning curves of the model during training.
    """
    def on_train_begin(self, logs={}):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []


    def on_epoch_end(self, epoch, logs={}):
        # Storing metrics
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]

        # Plotting
        metrics = [x for x in logs if 'val' not in x]

        f, axs = plt.subplots(1, len(metrics), figsize=(15,5))
        clear_output(wait=True)

        for i, metric in enumerate(metrics):
            axs[i].plot(range(1, epoch + 2),
                        self.metrics[metric],
                        label=metric, color='red')
            if logs['val_' + metric]:
                axs[i].plot(range(1, epoch + 2),
                            self.metrics['val_' + metric],
                            label='val_' + metric)

            axs[i].legend()
            axs[i].grid()

        plt.tight_layout()
        plt.show()


# In[36]:


callbacks_list = [PlotLearning(), TensorBoard(log_dir=output_dir)]
#callbacks_list = [TensorBoard(log_dir=output_dir)]

# Build the model
utils.set_random_seed(812)
epoch = 100
model = Sequential([
    Dense(units=4, activation='relu', input_shape = (X_train.shape[1],)),
    Dense(units=7, activation='softmax')
    ])
#Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model and capture the training history
history = model.fit(X_train, y_train, epochs=epoch, validation_data=(X_val, y_val), batch_size=32, callbacks=callbacks_list)


# In[37]:


callbacks_list = [PlotLearning(), TensorBoard(log_dir=output_dir)]
#callbacks_list = [TensorBoard(log_dir=output_dir)]

# Build the model
utils.set_random_seed(812)
epoch = 100
model = Sequential([
    Dense(units=4, activation='relu', input_shape = (X_train.shape[1],)),
    Dense(units=7, activation='softmax')
    ])
#Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model and capture the training history
history = model.fit(X_train, y_train, epochs=epoch, validation_data=(X_val, y_val), batch_size=32, callbacks=callbacks_list)


# In[133]:


callbacks_list = [PlotLearning(), TensorBoard(log_dir=output_dir)]
#callbacks_list = [TensorBoard(log_dir=output_dir)]

# Build the model
utils.set_random_seed(812)
epoch = 100
model = Sequential([
    Dense(units=16, activation='relu', input_shape = (X_train.shape[1],)),
    Dense(units=7, activation='softmax')
    ])
#Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model and capture the training history
history = model.fit(X_train, y_train, epochs=epoch, validation_data=(X_val, y_val), batch_size=32, callbacks=callbacks_list)


# In[134]:


callbacks_list = [PlotLearning(), TensorBoard(log_dir=output_dir)]
#callbacks_list = [TensorBoard(log_dir=output_dir)]

# Build the model
utils.set_random_seed(812)
epoch = 100
model = Sequential([
    Dense(units=32, activation='relu', input_shape = (X_train.shape[1],)),
    Dense(units=7, activation='softmax')
    ])
#Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model and capture the training history
history = model.fit(X_train, y_train, epochs=epoch, validation_data=(X_val, y_val), batch_size=32, callbacks=callbacks_list)


# In[27]:


loss, accuracy = model.evaluate(X_test, y_test)


# In[25]:


callbacks_list = [PlotLearning(), TensorBoard(log_dir=output_dir)]
#callbacks_list = [TensorBoard(log_dir=output_dir)]

# Build the model
utils.set_random_seed(812)
epoch = 100
model = Sequential([
    Dense(units=50, activation='relu', input_shape = (X_train.shape[1],)),
    Dense(units=7, activation='softmax')
    ])
#Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model and capture the training history
history = model.fit(X_train, y_train, epochs=epoch, validation_data=(X_val, y_val), batch_size=32, callbacks=callbacks_list)


# In[136]:


callbacks_list = [PlotLearning(), TensorBoard(log_dir=output_dir)]
#callbacks_list = [TensorBoard(log_dir=output_dir)]

# Build the model
utils.set_random_seed(812)
epoch = 100
model = Sequential([
    Dense(units=16, activation='relu', input_shape = (X_train.shape[1],)),
    Dense(units=10, activation='relu'),
    Dense(units=7, activation='softmax')
    ])
#Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model and capture the training history
history = model.fit(X_train, y_train, epochs=epoch, validation_data=(X_val, y_val), batch_size=32, callbacks=callbacks_list)


# In[139]:


callbacks_list = [PlotLearning(), TensorBoard(log_dir=output_dir)]
#callbacks_list = [TensorBoard(log_dir=output_dir)]

# Build the model
utils.set_random_seed(812)
epoch = 100
model = Sequential([
    Dense(units=32, activation='relu', input_shape = (X_train.shape[1],)),
    Dense(units=7, activation='softmax')
    ])
#Compile the model
sgd_optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model and capture the training history
history = model.fit(X_train, y_train, epochs=epoch, validation_data=(X_val, y_val), batch_size=32, callbacks=callbacks_list)


# In[28]:


# Get the 20,40,60,80 % of training data and plot.
def create_subsets():
    subsets = []
    fractions = [0.2, 0.4, 0.6, 0.8]
    n_samples = len(X_train)
    subsets = []
    for frac in fractions:
        print(f'frac = {frac}')
        size = int(frac * n_samples)
        indices = np.random.choice(len(X_train), size, replace=False)
        X_subset = X_train.iloc[indices]
        y_subset = y_train[indices]
        subsets.append((X_subset, y_subset))
    return subsets

subsets = create_subsets()
for subset in subsets:
    print(len(subset[0]), len(subset[1]))
subsets[0][1]


# In[10]:


# model
def model(X_train, y_train):
    callbacks_list = [PlotLearning(), TensorBoard(log_dir=output_dir)]
    #callbacks_list = [TensorBoard(log_dir=output_dir)]

    # Build the model
    utils.set_random_seed(812)
    epoch = 100
    model_obj = Sequential([
        Dense(units=50, activation='relu', input_shape = (X_train.shape[1],)),
        Dense(units=7, activation='softmax')
        ])
    #Compile the model
    model_obj.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model and capture the training history
    #history = model.fit(X_train, y_train, epochs=epoch, validation_data=(X_val, y_val), batch_size=32, callbacks=callbacks_list)
    history = model_obj.fit(X_train, y_train, epochs=epoch, validation_data=(X_val, y_val), batch_size=32)
    return history, model_obj


# In[11]:


history, model_obj = model(X_test, y_test)
loss, accuracy = model_obj.evaluate(X_test, y_test)


# In[ ]:





# In[30]:


histories = list()
for subset in subsets:
    history = model(subset[0], subset[1])
    histories.append(history)

    


# In[35]:


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))


axes[0,0].plot(histories[0].history['loss'],color='red',label='20%')
axes[0,0].plot(histories[1].history['loss'],color='orange',label='40%')
axes[0,0].plot(histories[2].history['loss'],color='magenta',label='60%')
axes[0,0].plot(histories[3].history['loss'],color='blue',label='80%')
axes[0,0].set_xlabel("Percentage of Data")
axes[0,0].set_ylabel("Loss")
axes[0,0].legend(loc="upper right")

axes[0,1].plot(histories[0].history['val_loss'],color='red',label='20%')
axes[0,1].plot(histories[1].history['val_loss'],color='orange',label='40%')
axes[0,1].plot(histories[2].history['val_loss'],color='magenta',label='60%')
axes[0,1].plot(histories[3].history['val_loss'],color='blue',label='80%')
axes[0,1].set_xlabel("Percentage of Data")
axes[0,1].set_ylabel("Validation Loss")
axes[0,1].legend(loc="upper right")

axes[1,0].plot(histories[0].history['accuracy'],color='red',label='20%')
axes[1,0].plot(histories[1].history['accuracy'],color='orange',label='40%')
axes[1,0].plot(histories[2].history['accuracy'],color='magenta',label='60%')
axes[1,0].plot(histories[3].history['accuracy'],color='blue',label='80%')
axes[1,0].set_xlabel("Percentage of Data")
axes[1,0].set_ylabel("Accuracy")
axes[1,0].legend(loc="upper right")

axes[1,1].plot(histories[0].history['accuracy'],color='red',label='20%')
axes[1,1].plot(histories[1].history['accuracy'],color='orange',label='40%')
axes[1,1].plot(histories[2].history['accuracy'],color='magenta',label='60%')
axes[1,1].plot(histories[3].history['accuracy'],color='blue',label='80%')
axes[1,1].set_xlabel("Percentage of Data")
axes[1,1].set_ylabel("Validation Accuracy")
axes[1,1].legend(loc="upper right")


# In[372]:


def get_accuracy(x_train, y_train, k=11):
    model=model(k)
    model.fit(x_train,y_train)
    prediction=model.predict(X_val)
    acc = accuracy_score(prediction,y_val)
    return acc


# In[393]:


acc_partial = []
for subset in subsets:
    X_train, y_train = subset
    acc_partial.append(get_accuracy(X_train, y_train))
plt.plot(fractions, acc_partial, color='blue')
#plt.xticks(k_range)
plt.xlabel("Fraction of training")
plt.ylabel("Accuracy")
#plt.show()


# In[158]:


print(len(X_train_20_sampled), len(y_train_20_sampled))
print(len(X_train_40_sampled), len(y_train_40_sampled))
print(len(X_train_60_sampled), len(y_train_60_sampled))
print(len(X_train_80_sampled), len(y_train_80_sampled))


# In[175]:


for i in range(len(y_train_sample)):
    print(get_accuracy(x_ss[i], y_train_sample[i]))


# In[159]:


print(get_accuracy(X_train_20_sampled, y_train_20_sampled))
print(get_accuracy(X_train_40_sampled, y_train_40_sampled))
print(get_accuracy(X_train_60_sampled, y_train_60_sampled))
print(get_accuracy(X_train_80_sampled, y_train_80_sampled))
'''
plt.plot(k_range, val_20, color='green',label='20')
plt.plot(k_range, val_40, color='blue',label='40')
plt.plot(k_range, val_60, color='red',label='60')
plt.plot(k_range, val_80, color='black',label='80')
plt.legend(loc='upper right')
plt.xticks(k_range)
plt.xlabel("Percentage")
plt.ylabel("Accuracy")
'''


# In[ ]:





# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# Assuming X_train and y_train are your features and labels for the training data

# Define the parameter grid for k values
param_grid = {'n_neighbors': range(1, 21)}  # Search for k from 1 to 20

for i in range(1, 100, 2):
    # Create a k-Nearest Neighbors classifier
    knn = KNeighborsClassifier()

    # Perform grid search to find the best k value
    grid_search.fit(X_train, y_train)

# Print the best k value and its corresponding accuracy score
print("Best k:", grid_search.best_params_['n_neighbors'])
print("Best accuracy:", grid_search.best_score_)


# In[68]:


from yellowbrick.model_selection import ValidationCurve
#cv = StratifiedKFold(4)
param_range = range(1,500,10)
oz = ValidationCurve(
    KNeighborsClassifier(), param_name="n_neighbors",
    #param_range=param_range, cv=cv, scoring="accuracy", n_jobs=4,
    param_range=param_range, scoring="accuracy",
)
oz.fit(X, y)
oz.show()


# In[63]:


from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# Assuming X_train and y_train are your features and labels for the training data

# Define the parameter grid for k values
param_grid = {'n_neighbors': range(1, 21)}  # Search for k from 1 to 20

# Create a k-Nearest Neighbors classifier
knn = KNeighborsClassifier()

# Create a GridSearchCV object
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')

# Perform grid search to find the best k value
grid_search.fit(X_train, y_train)

# Print the best k value and its corresponding accuracy score
print("Best k:", grid_search.best_params_['n_neighbors'])
print("Best accuracy:", grid_search.best_score_)


# In[62]:


# Create a k-Nearest Neighbors classifier
model = KNeighborsClassifier(n_neighbors=5)

# Create a LearningCurve visualizer
visualizer = LearningCurve(model, scoring='accuracy')

# Fit and visualize the learning curve
visualizer.fit(X_train, y_train)
visualizer.show()


# In[ ]:





# In[201]:


param_range = range(1,10302, 100)

plt.plot(param_range,result)
#plt.xticks(param_range)


# In[218]:


print(len(X_val))
print(len(y_val))
from collections import Counter
Counter(y_train)
#Counter(y_val)
prediction


# In[243]:


y_train_1 = to_categorical(y_train)
y_val_1 = to_categorical(y_val)


# In[245]:


utils.set_random_seed(812)
epoch = 500
#CHange the layers or neurons
model = Sequential([
    Dense(units=4, activation='relu', input_shape = (X_train.shape[1],)),
    Dense(units=5, activation='relu'),
    Dense(units=7, activation='softmax')
    ])
#Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model and capture the training history
#history = model.fit(X_train, y_train, epochs=epoch, validation_data=(X_val, y_val), batch_size=32)
history = model.fit(X_train, y_train_1, epochs=epoch, validation_data=(X_val, y_val_1))



# In[5]:


from keras.wrappers.scikit_learn import KerasClassifier

# Define the Keras model
def create_model(optimizer='adam', activation='relu', neurons=16):
    model = Sequential()
    model.add(Dense(neurons, input_dim=4, activation=activation))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Wrap the Keras model with KerasClassifier
keras_classifier = KerasClassifier(build_fn=create_model, verbose=0)

# Define the parameter grid
param_grid = {
    'optimizer': ['adam', 'rmsprop'],
    'activation': ['relu', 'tanh'],
    'neurons': [8, 16, 32]
}

# Use GridSearchCV
grid_search = GridSearchCV(keras_classifier, param_grid, cv=3, scoring='accuracy')

# Fit the GridSearchCV object to your data
grid_search.fit(X_train, y_train)

# Retrieve the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best parameters:", best_params)
print("Best score:", best_score)


# In[ ]:




