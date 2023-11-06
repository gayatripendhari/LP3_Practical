import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("C:\\Users\\Admin\\Downloads\\ML Codes\\ML Codes\\Churn_Modelling.csv")
data.head()

data.shape

data.isnull().sum()

for col in data.columns:
    if data[col].dtype != 'object':
        bp = sns.boxplot(data = data, x = col)
        plt.show()
        sp = sns.scatterplot(data = data, x = col, y=data["Exited"])
        plt.show()

bp = sns.boxplot(data = data, x = 'Age')
plt.show()
sp = sns.scatterplot(data = data, x = 'Age', y=data["Exited"])
plt.show()

Q1_age = data['Age'].quantile(0.25)
Q3_age = data['Age'].quantile(0.75)
IQR_age = Q3_age - Q1_age
data.drop(data[data['Age']>83].index, axis = 0, inplace =True)

bp = sns.boxplot(data = data, x = 'Age')
plt.show()
sp = sns.scatterplot(data = data, x = 'Age', y=data["Exited"])
plt.show()

data.shape

bp = sns.boxplot(data = data, x = 'NumOfProducts')
plt.show()
sp = sns.scatterplot(data = data, x = 'NumOfProducts', y=data["Exited"])
plt.show()

data.drop(data[data['NumOfProducts']>=4].index, axis = 0, inplace =True)

bp = sns.boxplot(data = data, x = 'NumOfProducts')
plt.show()
sp = sns.scatterplot(data = data, x = 'NumOfProducts', y=data["Exited"])
plt.show()

data.shape

sns.countplot(x="Exited", data=data)

geography = pd.get_dummies(data['Geography'], drop_first=True)
gender = pd.get_dummies(data['Gender'], drop_first=True)

data = pd.concat([data, geography, gender], axis=1)
data.info()

X = data.drop(['Exited', 'CustomerId', 'Surname', 'RowNumber', 'Geography', 'Gender'], axis=1)
Y = data['Exited']
X.shape

from imblearn.over_sampling  import RandomOverSampler
smoteOver = RandomOverSampler(sampling_strategy=1)
X, Y = smoteOver.fit_resample(X,Y)

X.shape

from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size= 0.20, random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_Train = sc.fit_transform(X_Train)
X_Test = sc.transform(X_Test)

import keras
from keras.models import Sequential
from keras.layers import Dense

X_Train.shape

model = Sequential()

model.add(Dense(units=6, kernel_initializer='he_uniform', activation='relu', input_dim=11))
model.add(Dense(units=6, kernel_initializer='he_uniform', activation='relu'))
model.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the Training set
model_history = model.fit(X_Train, Y_Train, batch_size=10, validation_split=0.33, epochs=100)


# Summarize history for loss
plt.figure(figsize=(8,8))
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

# Summarize history for accuracy
plt.figure(figsize=(8,8))
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower right')
plt.show()

Y_Pred = model.predict(X_Test)
Y_Pred = (Y_Pred > 0.5)
Y_Pred

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm = confusion_matrix(Y_Test, Y_Pred)

def cmatrix_fun(model_name, actual, predicted):
    # check the confusion matrix
    cm = confusion_matrix(actual, predicted) 
    print(cm)

    # Plot the CM
    ax = sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap='Blues')

    ax.set_title(f'The confusion matrix using {model_name} Classifier \n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');

    # Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])

cmatrix_fun('ANN model', Y_Test, Y_Pred)

print(classification_report(Y_Test, Y_Pred))

