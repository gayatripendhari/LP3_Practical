import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score 
from sklearn.metrics import confusion_matrix, accuracy_score 
from sklearn.metrics import classification_report 
 

df=pd.read_csv("emails.csv") 


print(df.head() )
print(df.info())
df.isnull().sum() 


X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101) 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred) 
cm


cl_report=classification_report(y_test,y_pred)
print(cl_report) 
print("Accuracy Score for KNN : ", accuracy_score(y_pred,y_test))



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix, accuracy_score 
from sklearn.metrics import classification_report



df=pd.read_csv("C:\\Users\\Admin\\Downloads\\emails.csv")
print(df.head() )
print(df.info())
print(df.isnull().sum() )

 
 
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values 
X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101) 
svc = SVC(C=1.0,kernel='rbf',gamma='auto')
svc.fit(X_train,y_train)
y_pred2 = svc.predict(X_test) 
cm = confusion_matrix(y_test, y_pred2) 
cm




print("Accuracy Score for SVC : ", accuracy_score(y_pred2,y_test)) 
cl_report=classification_report(y_test,y_pred2)
print(cl_report)