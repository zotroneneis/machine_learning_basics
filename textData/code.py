import pandas as pd
import sklearn
from sklearn import preprocessing
Encoder = preprocessing.LabelEncoder()
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
vect = CountVectorizer()

s1 = pd.read_csv('Sheet_1.csv',encoding='latin-1')
s2 = pd.read_csv('Sheet_2.csv',encoding='latin-1')

s1['Label'] = Encoder.fit_transform(s1['class'])
x = s1.response_text
y = s1.Label

Xtrain, Xtest, Ytrain,Ytest = train_test_split(x,y,random_state=0)
Xtrain = vect.fit_transform(Xtrain)
Xtest = vect.transform(Xtest)
model = GradientBoostingClassifier(max_depth=20, max_features=10)
model.fit(Xtrain,Ytrain)
print('for sheet 1 is : ' + str(sklearn.metrics.accuracy_score(Ytest,model.predict(Xtest))))

s2['Label'] = Encoder.fit_transform(s2['class'])
x = s2.resume_text
y = s2.Label

Xtrain, Xtest, Ytrain,Ytest = train_test_split(x,y,random_state=0)
Xtrain = vect.fit_transform(Xtrain)
Xtest = vect.transform(Xtest)
model = GradientBoostingClassifier(max_depth=20, max_features=10)
model.fit(Xtrain,Ytrain)
print('for sheet 2 is : ' + str(sklearn.metrics.accuracy_score(Ytest,model.predict(Xtest))))