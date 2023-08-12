import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

df=pd.read_csv('diabetes.csv')

X=df.iloc[:,:-1]
y=df.iloc[:,-1]

classifier=RandomForestClassifier()
classifier.fit(X,y)

pickle_out = open("diabeticmodel.pkl","wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()