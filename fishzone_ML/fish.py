import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.metrics import precision_recall_curve

np.random.seed(40)
td = pd.read_csv('sample_nan.csv')


#'''
#Sea Surface Temperature, Chlorophyll,
#Relative Humidity, Sea Level Pressure, Air Temperature, Total Cloudiness and Total Fish catch data.
#'''

# Remove year and Month and Label
#print(td)

df =  td[['SST', 'SSC', 'AT', 'RH', 'SLP', 'TC', 'TOTALOIL']]

#print(df)
#Shuffle the dataset and apply interpolate method

df = df.sample(frac=1).reset_index(drop=True)
nedf = df.interpolate(method='cubic', axis=0).ffill().bfill()
nedf = nedf.astype("float")
ssc = np.array(nedf['SSC'])
sst = np.array(nedf['SST'])
fc = np.array(nedf['TOTALOIL'])
lab = []

for i in range(len(ssc)):
    if ssc[i]>0.2 and sst[i]>25.0 and fc[i]>10000:
        lab.append("PFZ")
    else:
        lab.append("NPFZ")

label = pd.DataFrame(lab, columns=['label'])
dataset = pd.concat([nedf,label],axis=1)


dataset.to_csv("cubic_interpolation.csv",sep='\t', encoding='utf-8')

# create a copy
df1 = dataset

#mapping
df1['label']=df1['label'].map({'PFZ':1,'NPFZ':0})

# Drop Total catch
df2=df1.drop(['TC'],axis=1)
X = df2[['SST', 'SSC', 'AT', 'RH', 'SLP', 'TOTALOIL']]

#print("x here\t",X)
Y = df2[['label']]

#print("\t",Y)
#preprocessing....
from sklearn import preprocessing
#X_norm = preprocessing.normalize(X, norm='l2')   x[test_ids[:-10]]

X_norm = X.values
print(X_norm)


#y = np.squeeze(np.array(Y).reshape(1,-1))
y = Y.values
print(y)

#print("\t",y)

#feature importance graph
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X_norm,y)
feature_importance=model.feature_importances_
print(model.feature_importances_)
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .8
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()

############################################################################################################
#taking random indices to split the dataset into train and test
test_ids = np.random.permutation(len(X_norm))
test_ids=test_ids.tolist()
test_ids.sort()
print(np.asarray(test_ids))

#splitting data and labels into train and test
#keeping last 10 entries for testing, rest for training

x_train = X_norm[test_ids[:-10]]

#reading the test file from test.csv
tdt = pd.read_csv('test.csv')
dft =  tdt[['SST', 'SSC', 'AT', 'RH', 'SLP',  'TOTALOIL']]






#x_test = X_norm[test_ids[-10:]]
print("xtest \t \t \t ",dft.values)
x_test=dft.values
y_train = y[test_ids[:-10]]

#y_test = y[test_ids[-3:]] use this only to find accuracy which are labels 1,0

#classifying using decision tree
clf = tree.DecisionTreeClassifier()

#training (fitting) the classifier with the training set
clf.fit(x_train, y_train)

#predictions on the test dataset
pred = clf.predict(x_test)
pred=pred.tolist()
print (pred) #predicted labels i.e 1 for pfz otherwise no

#now plotting only those places that were pfz


import folium
# Make a data frame with dots to show on the map
# data = pd.DataFrame({
#     'name': [],
#     'lat': [],
#     'lon': []
#
# })

dftmap =  tdt[['place','lat','lon']]
dftmap=dftmap.values.tolist()
#print(dftmap)
ind=0
maploc=[]
for i in dftmap:

    if pred[ind]==1:
        maploc.append(i)
    ind=ind+1

data=pd.DataFrame(maploc,columns=['name','lat','lon'],index=None)
print(data)

# Make an empty map
m = folium.Map(location=[20, 0], tiles="Mapbox Bright", zoom_start=2)

# I can add marker one by one on the map
for i in range(0, len(data)):
    folium.Marker([data.iloc[i]['lat'], data.iloc[i]['lon']], popup=data.iloc[i]['name']).add_to(m)

# Save it as html
m.save('keralafish.html')


#print (y_test) #actual labels that is if we knew

#print (accuracy_score(pred, y_test)) #to test accuracy prediction accuracy



###################################################################################################################
#applymachinelearning using randomforest
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# X_train, X_test, y_train, y_test = train_test_split(X_norm,y, test_size=0.33, random_state=40)
# #print("xtrain",X_train,"\nxtest",X_test,"\nytrain",y_test,"\nytest",y_test)
# clfRF = RandomForestClassifier(random_state=0)
# clfRF.fit(X_train, y_train)
# predRF = clfRF.predict(X_test)
#
# print(accuracy_score(predRF,y_test))
#
# print(confusion_matrix(y_test,predRF))