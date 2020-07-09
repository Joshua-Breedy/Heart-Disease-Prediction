import sklearn
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("processed.cleveland.data")
le = preprocessing.LabelEncoder()
age = le.fit_transform(list(data["age"]))
sex = le.fit_transform(list(data["sex"]))
cp = le.fit_transform(list(data["cp"]))
trestbps = le.fit_transform(list(data["trestbps"]))
chol = le.fit_transform(list(data["chol"]))
fbs = le.fit_transform(list(data["fbs"]))
restecg = le.fit_transform(list(data["restecg"]))
thalach = le.fit_transform(list(data["thalach"]))
exang = le.fit_transform(list(data["exang"]))
oldpeak = le.fit_transform(list(data["oldpeak"]))
slope = le.fit_transform(list(data["slope"]))
ca = le.fit_transform(list(data["ca"]))
thal = le.fit_transform(list(data["thal"]))
num = le.fit_transform(list(data["num"]))

predict = "num"
X = list(zip(age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal))
y = list(num)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

model = KNeighborsClassifier(n_neighbors=11,leaf_size=2,p=1)

model.fit(x_train, y_train)
acc = model.score(x_test,y_test)
print(acc)

'''
# TRAIN MODEL MULTIPLE TIMES FOR BEST SCORE(Tech with Tim Saving Model)
best = 0
for _ in range(20):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    model = KNeighborsClassifier(n_neighbors=11,leaf_size=2,p=1)

    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    print("Accuracy: " + str(acc))

    if acc > best:
        best = acc
        with open("heart.pickle", "wb") as f:
            pickle.dump(model, f)

# LOAD MODEL
pickle_in = open("heart.pickle", "rb")
model = pickle.load(pickle_in)
print(model)
'''
#Hyperparameter optimization (https://github.com/fi-nik/KNN-and-Tuning-Hyperparameters/blob/master/Adipta%20Martulandi%20-%20KNN%20%2B%20Tuning.ipynb)
'''
#List Hyperparameters
leaf_size = list(range(1,50))
n_neighbors = list(range(1,30))
p=[1,2]
#Hyperparameters set
hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
knn_2 = KNeighborsClassifier()

clf = GridSearchCV(knn_2, hyperparameters, cv=10)

best_model = clf.fit(X,y)

#Prints optimal results for hyperparameters
print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
print('Best p:', best_model.best_estimator_.get_params()['p'])
print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])

y_pred = best_model.predict(x_test)

print(classification_report(y_test, y_pred))
'''