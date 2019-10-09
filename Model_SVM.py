import numpy as np 
import pandas as pd
import matplotlib.pyplot as pyplot 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


SEED = 8888
np.random.seed(SEED)

columns = ["Reason_for_the_call","Customer_ID","Account_ID","Age","Date_and_Time"]
dataset = pd.read_csv('excel_cust_dataset.csv',names=columns)
print(dataset.columns)
# dataset.drop(, axis=1)
print(dataset)
df_rev = dataset
df_rev = df_rev.apply(LabelEncoder().fit_transform)

# print(df_rev)
features = df_rev.values[:,1:]
target = df_rev.values[:,0]

# columns1 = ["Customer_ID","Account_ID","Age","Date_and_Time"]
# scaled_features = {}
# for each in columns1:
#     mean , std = df_rev[each].mean(), df_rev[each].std()
#     scaled_features[each] = [mean,std]
#     df_rev.loc[:, each] = (df_rev[each]-mean)/std


features_train, features_test, target_train, target_test = train_test_split(features,target, test_size = 0.3,random_state = 10)

print(features_train)
print(target_train)
clf = SVC()
clf.fit(features_train,target_train)
target_pred = clf.predict(features_test)

print(accuracy_score(target_test, target_pred, normalize = True))