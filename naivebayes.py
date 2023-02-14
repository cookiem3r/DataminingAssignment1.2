from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import time
from sklearn.model_selection import GridSearchCV,RepeatedKFold

#Import CSV file containing data
dataset = pd.read_csv('data/adult.datawithheaders.csv')

#Remove ? for workclss
dataset = dataset[dataset.workclass != ' ?']

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
dataset.insert(1,"agerange",agerange)

#Hrs worked
hrsrange = pd.cut(dataset["hrs"],bins=[0,20,40,60,80,100],labels=["0","1","2","3","4"])
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
dataset["agerange"]=dataset["agerange"].astype("int")
dataset["capgainrange"]=dataset["capgainrange"].astype("int")
dataset["hrsrange"]=dataset["hrsrange"].astype("int")
dataset["fnlwgtrange"]=dataset["fnlwgtrange"].astype("int")

print(dataset.dtypes)

X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.5)

#GridSearch for hyper-parameters
def gridsearch():
    params = {'alpha': [0.01, 0.1, 0.5, 1.0, 10.0],
            'class_prior': [None, [0.1,] * 12, ],
            'fit_prior': [True, False]
            }
    cv = RepeatedKFold(n_splits=5, n_repeats=5)
    categorical_nb_grid = GridSearchCV(estimator=CategoricalNB(), param_grid=params, n_jobs=-1, cv=cv, verbose=5)
    categorical_nb_grid.fit(X_train,y_train)

    print('Best Parameters : {}'.format(categorical_nb_grid.best_params_))
    print('Best Accuracy Through Grid Search : {:.3f}\n'.format(categorical_nb_grid.best_score_))


#Grid Search to find optimal hyper-parameters
# Best Parameters : {'alpha': 10.0, 'class_prior': None, 'fit_prior': True}
#Best Accuracy Through Grid Search : 0.809
#gridsearch()

nb = CategoricalNB(alpha=0.5,fit_prior=True)

#Start Timer
trainstart = time.time()
#Train Model
classifier = nb.fit(X_train, y_train)
#Stop Timer
trainstop = time.time()
print(f"Training time: {trainstop - trainstart}s")

predictstart = time.time()
y_pred = classifier.predict(X_test)
predictstop = time.time()
print(f"Predict time: {predictstop - predictstart}s")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
