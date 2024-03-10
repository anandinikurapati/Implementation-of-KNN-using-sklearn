# Implementation-of-KNN-using-sklearn
# Datasetslink:https://drive.google.com/drive/folders/15XG8HzPdMaWgGYv5DGG4uN4KL00Nebt1?usp=share_link
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np
%matplotlib inline
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
df = pd.read_csv("ClassifiedData.csv",index_col=0) 
scaler = StandardScaler() 
scaler.fit(df.drop('TARGET CLASS',axis=1))
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1]) 
X_train,X_test,y_train, y_test = train_test_split(scaled_features,
df['TARGET CLASS'], test_size=0.30)
#Initially with K=1
knn1 = KNeighborsClassifier(n_neighbors=1) 
knn1.fit(X_train,y_train)
pred1 = knn1.predict(X_test) 
print("For K=1 results are:") 
print(confusion_matrix(y_test,pred1))
print(classification_report(y_test,pred1))
# NOW WITH K=23
knn23 = KNeighborsClassifier(n_neighbors=23)
knn23.fit(X_train,y_train) 
pred23 = knn23.predict(X_test)
print("For K=23 results are:") 
print(confusion_matrix(y_test,pred23)) 
print(classification_report(y_test,pred23)) 
