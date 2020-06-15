# -*- coding: utf-8 -*-
"""
Created on Sat May 16 15:51:04 2020

@author: kingslayer
"""

##### DATA PREPROCESSING #######


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer

cancer=load_breast_cancer()


#Creating the DataFrame
df_cancer=pd.DataFrame(np.c_[cancer["data"],cancer["target"]],columns=np.append(cancer["feature_names"],["target"]))



##### VISUALISING THE DATA #####

sns.pairplot(df_cancer,hue="target",vars=["mean radius","mean texture",'mean perimeter',"mean area"])

sns.countplot(df_cancer["target"])

sns.scatterplot(x=df_cancer["mean radius"],y=df_cancer['mean area'],hue=df_cancer["target"])




####### TRAINING THE MODEL ########
X=df_cancer.drop("target",axis=1)
y=df_cancer["target"]


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=5)


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


from sklearn.svm import SVC
classifier=SVC(kernel="rbf",C=10,gamma=0.001)
classifier.fit(X_train,y_train)


y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(classifier,X=X_train,y=y_train,cv=10,scoring="accuracy")
mean_accuracy=accuracies.mean()

from sklearn.model_selection import GridSearchCV
params=[{'C':[1.0,10.0,5.0],'kernel':['rbf'],'gamma':[1,0.1,0.001]}]
gridsearch=GridSearchCV(estimator=classifier,param_grid=params,scoring="accuracy",cv=10)
gridsearch=gridsearch.fit(X_train,y_train)
best_param=gridsearch.best_params_
best_score=gridsearch.best_score_


sns.heatmap(cm,annot=True)