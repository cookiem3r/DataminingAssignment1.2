import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import time

#Import CSV file containing data
dataset = pd.read_csv('data/adult.datawithheaders.csv')

#Remove ? for workclss
dataset = dataset[dataset.workclass != ' ?']
dataset.to_csv('dp.csv')

#Encode columns from string to int values using LabelEncoder
dataset.workclass = LabelEncoder().fit_transform(dataset.workclass)
dataset.maritalstatus = LabelEncoder().fit_transform(dataset.maritalstatus)
dataset.occupation = LabelEncoder().fit_transform(dataset.occupation)
dataset.relationship = LabelEncoder().fit_transform(dataset.relationship)
dataset.race = LabelEncoder().fit_transform(dataset.race)
dataset.sex = LabelEncoder().fit_transform(dataset.sex)
dataset.country = LabelEncoder().fit_transform(dataset.country)
dataset.pay = LabelEncoder().fit_transform(dataset.pay)

#Convert contineous data into discrete data (Range)
#Age
agerange = pd.cut(dataset["age"],bins=[0,17,30,40,50,60,99],labels=["0","1","2","3","4","5"])
dataset.insert(1,"Agerange",agerange)

#Hrs worked
hrsrange = pd.cut(dataset["hrs"],bins=[0,35,41,47,53,59,65,100],labels=["0","1","2","3","4","5","6"])
dataset.insert(14,"hrsrange",hrsrange)

#Capgain
capgain = pd.cut(dataset["capgain"],bins=[-1,4001,100000],labels=["0","1"])
dataset.insert(11,"capgainrange",capgain)

#fnlweight
fnlweight = pd.cut(dataset["fnlwgt"],bins=[-1,100000,200000,300000,400000,1500000],labels=["0","1","2","3","4"])
dataset.insert(2,"fnlwgtrange",fnlweight)

print(dataset.head(20))

#Get Label column
label = dataset.iloc[:,18]

print(dataset.head(10))
#drop column that are not needed
cols = [0,4,5,13,14,15,18]
dataset.drop(dataset.columns[cols],axis=1,inplace=True)

print(dataset.head(10))
print(label.head(10))

X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.4)

#Hyper-parameters
classifier = DecisionTreeClassifier(max_depth = 8, random_state = 0, criterion="gini", min_samples_leaf=30, ccp_alpha=0.002)

#Start Timer
trainstart = time.time()
#Train Model
classifier = classifier.fit(X_train, y_train)
#Stop Timer
trainstop = time.time()

print(f"Training time: {trainstop - trainstart}s")
print("Depth: %s" % (classifier.get_depth()))
print("Leaves: %s" % (classifier.get_n_leaves()))

#Cost Complexity Pruning Path function for ccp_alpha hyperparameters
def costcomplexity():
    path = classifier.cost_complexity_pruning_path(X_train,y_train)
    print(path['ccp_alphas'])

    accuracy_train,accuracy_test=[],[]
    for i in path["ccp_alphas"]:
        tree = DecisionTreeClassifier(ccp_alpha=i)
        tree.fit(X_train,y_train)
        y_train_pred=tree.predict(X_train)
        y_test_pred=tree.predict(X_test)
        accuracy_train.append(metrics.accuracy_score(y_train,y_train_pred))
        accuracy_test.append(metrics.accuracy_score(y_test,y_test_pred))

    sns.set()
    plt.figure(figsize=(14,7))
    sns.lineplot(y=accuracy_train,x=path['ccp_alphas'],label="Train Accuracy")
    sns.lineplot(y=accuracy_test,x=path['ccp_alphas'],label="Test Accuracy")
    plt.xticks(ticks=np.arange(0.00,0.07,0.001))
    plt.show()


#Prediction function
def predict():
    predictstart = time.time()
    y_pred = classifier.predict(X_test)
    predictstop = time.time()
    print(f"Predict time: {predictstop - predictstart}s")
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    plt.figure(figsize=(12,12))
    tree.plot_tree(classifier,fontsize=10)
    plt.show()

#Below function plots the accuracy graph of Training and Testing Accuracy against ccp_alpha
#Optimal ccp_alpha is around 0.001 to achieve an accuracy of around 83%
#costcomplexity()

predict()