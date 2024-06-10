#!/usr/bin/env python
# coding: utf-8

# In[36]:


import os
import warnings
import keras
import re


from time import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Import the libs for test split,KNN, svm and DecisionTree Classifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import learning_curve, validation_curve

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


# In[37]:


dry_bean = fetch_ucirepo(id=602)
# data (as pandas dataframes) 
X = dry_bean.data.features 
y = dry_bean.data.targets 
# metadata 
print(dry_bean.metadata) 

# variable information 
print(dry_bean.variables)


# In[38]:


df = pd.concat([X, y], axis=1)
df.head()


# In[39]:


feature_columns = df.columns.to_numpy()[:-1]
X = df[feature_columns]
y = df['Class']

# Label Encode for y
#lbl = LabelEncoder()
#y = lbl.fit_transform(y)
#y = to_categorical(y)


# In[40]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.2, random_state=142)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.2, random_state=42)


# In[41]:


sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_train = pd.DataFrame(X_train_scaled)

X_val_scaled = sc.fit_transform(X_val)
X_val = pd.DataFrame(X_val_scaled)



# In[42]:


fractions = np.arange(.2,.8,.05)
n_samples = len(X_train)
subsets = []

for frac in fractions:
    size = int(frac * n_samples)
    indices = np.random.choice(len(X_train), size, replace=False)    
    X_subset = X_train.iloc[indices]
    y_subset = y_train.iloc[indices]
    '''
    print(indices)
    print(X_train.index)
    print(len(X_train))
    '''
    #y_subset = y_train[indices]
    
    subsets.append((X_subset, y_subset))


# In[76]:


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


# In[312]:


'''
x_ss = []
sc = StandardScaler()
for x in X_train_sample:
    x_ss.append(pd.DataFrame(sc.fit_transform(x),index=x.index))
'''

    


# In[70]:


y_val


# In[43]:


# Create the output dir
output_dir = '/tmp/logs'
os.makedirs(output_dir, exist_ok=True)
os.path.exists(output_dir)


# In[44]:


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
        #y_subset = y_train[indices]
        y_subset = y_train.iloc[indices]
        subsets.append((X_subset, y_subset))
    return subsets

subsets = create_subsets()
for subset in subsets:
    print(len(subset[0]), len(subset[1]))
subsets[0][1]


# In[45]:


X_train


# In[50]:


# Train SVM models with different kernels
kernels = ['linear', 'rbf', 'poly']
models = []
for kernel in kernels:
    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)
    models.append((model, X_train, X_val, y_train, y_val, kernel))


# In[56]:


def bar_graph(score, score_type):
    # Data
    #categories = ['', 'BOMBAY', 'CALI', 'DERMASON', 'HOROZ', 'SEKER', 'SIRA', 'weighted avg']
    #colors = ['blue', 'green', 'red', 'purple', 'orange', 'yellow', 'cyan', 'black']
    kernels = ['linear', 'rbf', 'poly']
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(kernels, score, marker='o', linestyle='-', color='b')  # Line plot with markers
    plt.title('Model Performance')
    plt.xlabel('Model')
    plt.ylabel(f'{score_type}')
    plt.grid(True)
    plt.tight_layout()

    # Annotate each point
    for i, txt in enumerate(score):
        plt.annotate("{:.4f}".format(txt), (kernels[i], score[i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.show()


# In[57]:


# Generate evaluation metrics and plots
weighted_precision_values = []
weighted_recall_values = []
weighted_f1_values = []
accuracies = []
for i, (model, X_train, X_val, y_train, y_val, kernel) in enumerate(models):
    y_pred = model.predict(X_val)
    print(f"Kernel: {model.kernel}")
    report = classification_report(y_val, y_pred, output_dict=True)
    print(report)
    weighted_precision_values.append(report['weighted avg']['precision'])
    weighted_recall_values.append(report['weighted avg']['recall'])
    weighted_f1_values.append(report['weighted avg']['f1-score'])
    y_val_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)
    accuracies.append(accuracy)

    
print(weighted_precision_values)
print(weighted_recall_values)
print(weighted_f1_values)
bar_graph(weighted_precision_values, 'precisio')
bar_graph(weighted_recall_values, 'recall')
bar_graph(weighted_f1_values, 'f1-score')
bar_graph(accuracies, 'accuracy')
'''
    print(report)
    #sns.heatmap(precision, annot=True, cmap='Blues')
    
    
    #create_heatmap(report)
    #ax = axes[i]
    #plot_decision_boundary(model, X_train, y_train, ax)
'''


# In[22]:


# Learning curves
def plot_learning_curve(estimator, X, y, title, ax, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1,
                    color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color="g")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
            label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
            label="Cross-validation score")

    ax.set_title(title)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    ax.legend(loc="best")

fig, axes = plt.subplots(len(models), 1, figsize=(10, 20))

for i, (model, X_train, X_val, y_train, y_val, title) in enumerate(models):
    plot_learning_curve(model, X_train, y_train, f"Learning Curve: {title}, Kernel: {model.kernel}", axes[i])

plt.show()


# In[282]:


param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
}
# Validation curves
def plot_validation_curve(estimator, X, y, param_name, param_range, title, ax, cv=None, n_jobs=None):
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=cv, n_jobs=n_jobs)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    ax.fill_between(param_range, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1,
                    color="r")
    ax.fill_between(param_range, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color="g")
    ax.plot(param_range, train_scores_mean, 'o-', color="r",
            label="Training score")
    ax.plot(param_range, test_scores_mean, 'o-', color="g",
            label="Cross-validation score")

    ax.set_title(title)
    ax.set_xlabel("Parameter")
    ax.set_ylabel("Score")
    ax.legend(loc="best")

#fig, axes = plt.subplots(4, 2, figsize=(20, 18))
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

(_, X_train, X_val, y_train, y_val, title) = models[1]

#for i, (_, X_train, X_val, y_train, y_val, title) in enumerate(models):

  
plot_validation_curve(SVC(kernel='rbf'), X_train, y_train, 'gamma', param_grid['gamma'],
                        #f"Validation Curve: {title}, Kernel: RBF, Parameter: gamma", axes[i, 0], cv=5)
                        f"Validation Curve: {title}, Kernel: RBF, Parameter: gamma", axes[0], cv=5)

plot_validation_curve(SVC(kernel='rbf'), X_train, y_train, 'C', param_grid['C'],
                        f"Validation Curve: {title}, Kernel: RBF, Parameter: C", axes[1], cv=5)
'''
plot_validation_curve(SVC(kernel='rbf'), X_train, y_train, 'C', param_grid['C'],
                        f"Validation Curve: {title}, Kernel: RBF, Parameter: C", axes[i, 1], cv=5)

plot_validation_curve(SVC(kernel='rbf'), X_train, y_train, 'gamma', param_grid['gamma'],
                    f"Validation Curve: {title}, Kernel: RBF, Parameter: gamma", axes[i//2, i%2], cv=5)

plot_validation_curve(SVC(kernel='rbf'), X_train, y_train, 'C', param_grid['C'],
                    f"Validation Curve: {title}, Kernel: RBF, Parameter: C", axes[i//2, i%2], cv=5)
'''
plt.show()


# In[3]:


#Hyperparamter tuning using Gridsearch
from sklearn.model_selection import GridSearchCV


svm = SVC()
# param_grid = {'C':[0.01,0.05,0.1,1,10, 100, 1000],'kernel':['linear','rbf'], 'gamma':['scale','auto'] }
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf','linear']}
grid = GridSearchCV(svm,param_grid)


# In[23]:


#Calculating the accuracy of tuned model
grid_svc = grid.predict(X_test)
accuracy_score(y_val,grid_svc)


# In[5]:


def plot_validation_curve(estimator, X, y, param_name, param_range, title, ax, cv=None, n_jobs=None):
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=cv, n_jobs=n_jobs)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    ax.fill_between(param_range, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1,
                    color="r")
    ax.fill_between(param_range, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color="g")
    ax.plot(param_range, train_scores_mean, 'o-', color="r",
            label="Training score")
    ax.plot(param_range, test_scores_mean, 'o-', color="g",
            label="Cross-validation score")

    ax.set_title(title)
    ax.set_xlabel("Parameter")
    ax.set_ylabel("Score")
    ax.legend(loc="best")


# In[21]:


param_grid = {
    'C': [0.1, 1, 10],
    'degree': [2, 3, 4],
    'gamma': [0.1, 1, 10],
    'coef0': [0.0]#, 0.1, 1.0]
}
fig, axes = plt.subplots(1, 4, figsize=(10, 5))
plot_validation_curve(SVC(kernel='poly'), X_train, y_train, 'C', param_grid['C'],
                        #f"Validation Curve: {title}, Kernel: RBF, Parameter: gamma", axes[i, 0], cv=5)
                        f"Validation Curve: C, Kernel: RBF, Parameter: C", axes[0], cv=5)

plot_validation_curve(SVC(kernel='poly'), X_train, y_train, 'degree', param_grid['degree'],
                        #f"Validation Curve: {title}, Kernel: RBF, Parameter: degree", axes[1], cv=5)
                        f"Validation Curve: degree, Kernel: RBF, Parameter: degree", axes[1], cv=5)


plot_validation_curve(SVC(kernel='poly'), X_train, y_train, 'gamma', param_grid['gamma'],
                        #f"Validation Curve: {title}, Kernel: RBF, Parameter: gamma", axes[2], cv=5)
                        f"Validation Curve: gamma, Kernel: RBF, Parameter: gamma", axes[2], cv=5)

'''
plot_validation_curve(SVC(kernel='poly'), X_train, y_train, 'coef0', param_grid['coef0'],
                        f"Validation Curve: {title}, Kernel: RBF, Parameter: coef0", axes[3], cv=5)

'''


# In[152]:


histories = list()
for subset in subsets:
    history = model(subset[0], subset[1])
    histories.append(history)

    


# In[179]:


fig, axes = plt.subplots(nrows=2, ncols=2)


axes[0,0].plot(histories[0].history['loss'],color='red',label='20%')
axes[0,0].plot(histories[1].history['loss'],color='orange',label='40%')
axes[0,0].plot(histories[2].history['loss'],color='magenta',label='60%')
axes[0,0].plot(histories[3].history['loss'],color='blue',label='80%')
axes[0,0].xlabel("Percentage of Data")
axes[0,0].ylegend(loc="Loss")
axes[0,0].legend(loc="upper right")

axes[0,1].plot(histories[0].history['val_loss'],color='red',label='20%')
axes[0,1].plot(histories[1].history['val_loss'],color='orange',label='40%')
axes[0,1].plot(histories[2].history['val_loss'],color='magenta',label='60%')
axes[0,1].plot(histories[3].history['val_loss'],color='blue',label='80%')
axes[0,1].xlabel("Percentage of Data")
axes[0,1].ylegend(loc="Validation Loss")
axes[0,1].legend(loc="upper right")

axes[1,0].plot(histories[0].history['accuracy'],color='red',label='20%')
axes[1,0].plot(histories[1].history['accuracy'],color='orange',label='40%')
axes[1,0].plot(histories[2].history['accuracy'],color='magenta',label='60%')
axes[1,0].plot(histories[3].history['accuracy'],color='blue',label='80%')
axes[1,0].xlabel("Percentage of Data")
axes[1,0].ylegend(loc="Accuracy")
axes[1,0].legend(loc="upper right")

axes[1,1].plot(histories[0].history['accuracy'],color='red',label='20%')
axes[1,1].plot(histories[1].history['accuracy'],color='orange',label='40%')
axes[1,1].plot(histories[2].history['accuracy'],color='magenta',label='60%')
axes[1,1].plot(histories[3].history['accuracy'],color='blue',label='80%')
axes[0,0].xlabel("Percentage of Data")
axes[0,0].ylegend(loc="Validation Accuracy")
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



# In[ ]:





# In[ ]:




