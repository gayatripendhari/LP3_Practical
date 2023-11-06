import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("C:\\Users\\Hp\\OneDrive\\Documents\\BE ML Prac\\emails.csv")
df.head()


df.info()


df.isnull().sum()


X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)



from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
cm

from sklearn.metrics import classification_report
cl_report=classification_report(y_test,y_pred)
print(cl_report)


print("Accuracy Score for KNN : ", accuracy_score(y_pred,y_test))