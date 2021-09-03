import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle



read_file= pd.read_excel('Digital MQL Conversion 2019-21.xlsx',sheet_name='MQL raw data')
read_file.to_csv('Test.csv',index='None',header='True')
df = pd.DataFrame(pd.read_csv("Test.csv"))
df.head()

#Drop columns 
df.drop(['Unnamed: 0','Lead Owner','Last Activity','First Name','Last Name','Email','Title','Designation Level','Employee Range','Revenue Range','Create Date','Phone','MQL Solution','State','Appointment Set By','Onsite Appointment Set-for','MQL Date','MQL Campaign Name','Num of Opp','Num of Closed Won Deals','Closed Won Deals Amount'],axis=1,inplace=True)

df.drop([' Opp Amount'],axis=1,inplace=True)

#filling na values with the Most occurring values : Assumption
df['Lead Type'] = df['Lead Type'].fillna(df['Lead Type'].mode()[0])
df['Company / Account'] = df['Company / Account'].fillna(df['Company / Account'].mode()[0])
df['Lead Source'] = df['Lead Source'].fillna(df['Lead Source'].mode()[0])
df['Lead Status'] = df['Lead Status'].fillna(df['Lead Status'].mode()[0])
df['MQL Medium'] = df['MQL Medium'].fillna(df['MQL Medium'].mode()[0])
df['Industry'] = df['Industry'].fillna(df['Industry'].mode()[0])
df['Line of Services'] = df['Line of Services'].fillna(df['Line of Services'].mode()[0])
df['Lead Channel'] = df['Lead Channel'].fillna(df['Lead Channel'].mode()[0])
df['Marketing Subject'] = df['Marketing Subject'].fillna(df['Marketing Subject'].mode()[0])
df['Marketing Campaign Name'] = df['Marketing Campaign Name'].fillna(df['Marketing Campaign Name'].mode()[0])
df['MQL Subject'] = df['MQL Subject'].fillna(df['MQL Subject'].mode()[0])
df['MQL Services'] = df['MQL Services'].fillna(df['MQL Services'].mode()[0])
df['MQL Channel'] = df['MQL Channel'].fillna(df['MQL Channel'].mode()[0])
df['Converted to QAL'] = df['Converted to QAL'].fillna(0)
df['Converted to SAL'] = df['Converted to SAL'].fillna(0)

objFeatures = df.select_dtypes(include="object").columns

len(objFeatures)
#Iterate a loop for features of type object
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

for feat in objFeatures:
    df[feat] = le.fit_transform(df[feat].astype(str))

y = df.iloc[: , -2:]
x = df.iloc[:,0:-2]

from sklearn.model_selection import train_test_split
x_train, x_test , y_train, y_test= train_test_split(x,y,test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=5,criterion='entropy',random_state=0)
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

importances = list(classifier.feature_importances_)
feature_list = list(df.columns)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

preds = np.stack([t.predict(x_test) for t in classifier.estimators_])
np.mean(preds[:,0]), np.std(preds[:,0])

from sklearn import tree
plt.figure(figsize=(20,20))
_ = tree.plot_tree(classifier.estimators_[0], feature_names=x.columns, filled=True)

pickle.dump(classifier,open('model.pkl','wb'))
loaded_model=pickle.load(open('model.pkl','rb'))
result=loaded_model.score(x_test,y_test)
print(result)
