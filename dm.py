# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import optimizers
import pickle

from google.colab import drive
drive.mount('/content/drive')

"""Data Set : https://www.kaggle.com/datasets/massinissatinouche/electric-vehicle-population-2023"""

veri = pd.read_csv("/content/drive/MyDrive/data/evp.csv")

veri

plt.hist(veri["County"].value_counts())
plt.title("Vilayet Sayilari")
plt.show()

plt.hist(veri["City"].value_counts())
plt.title("Sehir Sayilari")
plt.show()

plt.hist(veri["State"].value_counts())
plt.title("Eyalet Sayilari")
plt.show()

plt.hist(veri["Make"].value_counts())
plt.title("Marka Sayilari")
plt.show()

plt.hist(veri["Model"].value_counts())
plt.title("Model Sayilari")
plt.show()

veri.columns

veri.drop(columns=["Electric Vehicle Type","Clean Alternative Fuel Vehicle (CAFV) Eligibility", "Electric Range", "Base MSRP", 
                   "Legislative District", "DOL Vehicle ID", "Vehicle Location", "Electric Utility", "2020 Census Tract", "Postal Code",
                   "VIN (1-10)"
                  ], inplace=True)

veri.head()

print("Farkli Vilayet sayilari = ",veri.County.nunique())
print("Farkli Sehir sayilari = ",veri.City.nunique())
print("Farkli Eyalet sayilari = ",veri.State.nunique())

print(veri.isnull().sum())

print(veri.isna().sum())

print(veri.isnull().sum().sum())
print(veri.isna().sum().sum())

veri.dropna(inplace=True)

print(veri.isnull().sum().sum())
print(veri.isna().sum().sum())

lveri = veri.apply(preprocessing.LabelEncoder().fit_transform)
print(lveri.head())

sns.heatmap(lveri.corr(), annot=True)

vilayet = veri.iloc[:,0:1]
le = preprocessing.LabelEncoder()
vilayet.iloc[:,0] = le.fit_transform(vilayet.iloc[:,0:1])
print(vilayet)

eyalet = veri.iloc[:,2:3]
le = preprocessing.LabelEncoder()
eyalet.iloc[:,0] = le.fit_transform(eyalet.iloc[:,0:1])
print(eyalet)

model_yil = veri.iloc[:,3:4]
print(model_yil)

marka = veri.iloc[:,4:5]
le = preprocessing.LabelEncoder()
marka.iloc[:,0] = le.fit_transform(marka.iloc[:,0:1])
print(marka)

model = veri.iloc[:,5:]
le = preprocessing.LabelEncoder()
model.iloc[:,0] = le.fit_transform(model.iloc[:,0:1])
print(model)

ohe = preprocessing.OneHotEncoder()
sehir = veri.iloc[:,1:2].values
sehir = ohe.fit_transform(sehir).toarray()
sehir

bagimli = enumerate(veri.iloc[:,1:2].values)
sonuc = np.zeros((len(veri.iloc[:,1:2].values), veri.City.nunique()))
sehirler = list(veri.City.unique())
val = list()
for i,j in bagimli:
    val.append((i,j[0]))

for i,j in val:
  sonuc[i,sehirler.index(j)] = 1.0

dfCity = pd.DataFrame(data=sonuc, columns = veri.City.unique())
dfCity

tyveri = veri.copy()
for i in sehirler:
  tyveri[i] = dfCity[i]

print(tyveri.isnull().sum().sum())
print(tyveri.isna().sum().sum())

totyveri = pd.concat([veri, dfCity], axis=1)

print(totyveri.isnull().sum().sum())
print(totyveri.isna().sum().sum())

yveri = veri.copy()
yveri = np.concatenate((veri,vilayet), axis=1)
yveri = np.concatenate((yveri,eyalet), axis=1)
yveri = np.concatenate((yveri,model_yil), axis=1)
yveri = np.concatenate((yveri,marka), axis=1)
yveri = np.concatenate((yveri,model), axis=1)
yveri = np.concatenate((yveri,sonuc), axis=1)

yveri = pd.DataFrame(yveri)

print(yveri.isnull().sum().sum())
print(yveri.isna().sum().sum())

veri.head()

yveri.head()

yveri.drop([0,1,2,3,4,5], axis =1, inplace=True)

yveri.rename(columns={6:"County", 7:"State", 8:"Model_Year", 9: "Make", 10:"Model"}, inplace=True)

i = 11
for j in sehirler:
  yveri.rename(columns={i:j}, inplace=True)
  i += 1

veri.head()

yveri.head()

for i in yveri.columns:
  yveri[i] = yveri[i].astype(int)

print(yveri.isnull().sum().sum())
print(yveri.isna().sum().sum())

yveri.to_csv("data.csv")

yveri = pd.read_csv("/content/drive/MyDrive/data/data.csv")

yveri.drop(columns=["Unnamed: 0"], inplace=True)

yveri.info(verbose=True, show_counts=True)

yveri.head()

yveri.describe()

X = yveri.iloc[:,:5]
X

y = yveri.iloc[:,5:]
y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = Sequential()
clf.add(Dense(units= 64, activation= "relu"))
clf.add(Dropout(0.5))
clf.add(Dense(units= 64, activation= "relu"))
clf.add(Dropout(0.5))
clf.add(Dense(units= 1, activation= "sigmoid"))

clf.compile(loss='mse', optimizer='adam', metrics= ["accuracy"])

clf.fit(X_train, y_train, epochs=10)

clf = Sequential()
clf.add(Dense(units= 64, activation= "relu"))
clf.add(Dropout(0.5))
clf.add(Dense(units= 64, activation= "relu"))
clf.add(Dropout(0.5))
clf.add(Dense(units= 1, activation= "softmax"))

clf.compile(loss='mse', optimizer='adam', metrics= ["accuracy"])

clf.fit(X_train, y_train, epochs=10)

clf = Sequential()
clf.add(Dense(units= 64, activation= "tanh"))
clf.add(Dropout(0.5))
clf.add(Dense(units= 64, activation= "tanh"))
clf.add(Dropout(0.5))
clf.add(Dense(units= 1, activation= "sigmoid"))

clf.compile(loss='mse', optimizer='adam', metrics= ["accuracy"])

clf.fit(X_train, y_train, epochs=10)

clf = Sequential()
clf.add(Dense(units= 64, activation= "softmax"))
clf.add(Dropout(0.5))
clf.add(Dense(units= 64, activation= "softmax"))
clf.add(Dropout(0.5))
clf.add(Dense(units= 1, activation= "sigmoid"))

clf.compile(loss='mse', optimizer='adam', metrics= ["accuracy"])

clf.fit(X_train, y_train, epochs=10)

clf = Sequential()
clf.add(Dense(units= 64, activation= "tanh"))
clf.add(Dropout(0.5))
clf.add(Dense(units= 64, activation= "tanh"))
clf.add(Dropout(0.5))
clf.add(Dense(units= 1, activation= "sigmoid"))

clf.compile(loss='mae', optimizer='adam', metrics= ["accuracy"])

clf.fit(X_train, y_train, epochs=10)

clf = Sequential()
clf.add(Dense(units= 64, activation= "relu"))
clf.add(Dropout(0.5))
clf.add(Dense(units= 64, activation= "relu"))
clf.add(Dropout(0.5))
clf.add(Dense(units= 1, activation= "sigmoid"))

clf.compile(loss='mae', optimizer='adam', metrics= ["accuracy"])

clf.fit(X_train, y_train, epochs=10)

clf = Sequential()
clf.add(Dense(units= 64, activation= "relu"))
clf.add(Dropout(0.5))
clf.add(Dense(units= 64, activation= "relu"))
clf.add(Dropout(0.5))
clf.add(Dense(units= 1, activation= "sigmoid"))

clf.compile(loss='mse', optimizer='adam', metrics= ["mae"])

clf.fit(X_train, y_train, epochs=10)

clf = Sequential()
clf.add(Dense(units= 64, activation= "tanh"))
clf.add(Dropout(0.5))
clf.add(Dense(units= 64, activation= "tanh"))
clf.add(Dropout(0.5))
clf.add(Dense(units= 1, activation= "sigmoid"))

clf.compile(loss='mse', optimizer='adam', metrics= ["mae"])

clf.fit(X_train, y_train, epochs=10)

clf = Sequential()
clf.add(Dense(units= 64, activation= "tanh"))
clf.add(Dropout(0.5))
clf.add(Dense(units= 64, activation= "tanh"))
clf.add(Dropout(0.5))
clf.add(Dense(units= 1, activation= "sigmoid"))

clf.compile(loss='mae', optimizer='adam', metrics= ["mse"])

clf.fit(X_train, y_train, epochs=10)

clf = Sequential()
clf.add(Dense(units= 64, activation= "relu"))
clf.add(Dropout(0.5))
clf.add(Dense(units= 64, activation= "relu"))
clf.add(Dropout(0.5))
clf.add(Dense(units= 1, activation= "sigmoid"))

clf.compile(loss='mae', optimizer='adam', metrics= ["mse"])

clf.fit(X_train, y_train, epochs=10)

"""Cok sinif"""

clf = Sequential()
clf.add(Dense(units= 64, activation= "relu", input_dim= 5))
clf.add(Dropout(0.5))
clf.add(Dense(units= 64, activation= "relu"))
clf.add(Dropout(0.5))
clf.add(Dense(units= 651, activation= "sigmoid"))

clf.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics= ["accuracy"])

clf.fit(X_train, y_train, epochs=1000,batch_size=512, validation_split=0.1)

clf = Sequential()
clf.add(Dense(units= 64, activation= "relu", input_dim= 5))
clf.add(Dropout(0.5))
clf.add(Dense(units= 64, activation= "relu"))
clf.add(Dropout(0.5))
clf.add(Dense(units= 651, activation= "sigmoid"))

clf.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics= ["accuracy"])

clf.fit(X_train, y_train, epochs=1000,batch_size=512, validation_split=0.1)

clf = Sequential()
clf.add(Dense(units= 64, activation= "relu", input_dim= 5))
clf.add(Dropout(0.5))
clf.add(Dense(units= 64, activation= "relu"))
clf.add(Dropout(0.5))
clf.add(Dense(units= 651, activation= "sigmoid"))

clf.compile(loss='mse', optimizer='rmsprop', metrics= ["accuracy"])

clf.fit(X_train, y_train, epochs=1000,batch_size=512, validation_split=0.1)

clf = Sequential()
clf.add(Dense(units= 64, activation= "relu", input_dim= 5))
clf.add(Dropout(0.5))
clf.add(Dense(units= 64, activation= "relu"))
clf.add(Dropout(0.5))
clf.add(Dense(units= 651, activation= "sigmoid"))

clf.compile(loss='mae', optimizer='rmsprop', metrics= ["accuracy"])

clf.fit(X_train, y_train, epochs=1000,batch_size=512, validation_split=0.1)

clf = Sequential()
clf.add(Dense(units= 64, activation= "tanh", input_dim= 5))
clf.add(Dropout(0.5))
clf.add(Dense(units= 64, activation= "tanh"))
clf.add(Dropout(0.5))
clf.add(Dense(units= 651, activation= "sigmoid"))

clf.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics= ["accuracy"])

clf.fit(X_train, y_train, epochs=1000,batch_size=512, validation_split=0.1)

clf = Sequential()
clf.add(Dense(units= 64, activation= "tanh", input_dim= 5))
clf.add(Dropout(0.5))
clf.add(Dense(units= 64, activation= "tanh"))
clf.add(Dropout(0.5))
clf.add(Dense(units= 651, activation= "sigmoid"))

clf.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics= ["accuracy"])

clf.fit(X_train, y_train, epochs=1000,batch_size=512, validation_split=0.1)

clf = Sequential()
clf.add(Dense(units= 64, activation= "tanh", input_dim= 5))
clf.add(Dropout(0.5))
clf.add(Dense(units= 64, activation= "tanh"))
clf.add(Dropout(0.5))
clf.add(Dense(units= 651, activation= "sigmoid"))

clf.compile(loss='mse', optimizer='rmsprop', metrics= ["accuracy"])

clf.fit(X_train, y_train, epochs=1000,batch_size=512, validation_split=0.1)

clf = Sequential()
clf.add(Dense(units= 64, activation= "tanh", input_dim= 5))
clf.add(Dropout(0.5))
clf.add(Dense(units= 64, activation= "tanh"))
clf.add(Dropout(0.5))
clf.add(Dense(units= 651, activation= "sigmoid"))

clf.compile(loss='mae', optimizer='rmsprop', metrics= ["accuracy"])

clf.fit(X_train, y_train, epochs=1000,batch_size=512, validation_split=0.1)

clf = Sequential()
clf.add(Dense(units= 64, activation= "relu", input_dim= 5))
clf.add(Dropout(0.5))
clf.add(Dense(units= 64, activation= "relu"))
clf.add(Dropout(0.5))
clf.add(Dense(units= 651, activation= "sigmoid"))

clf.compile(loss='binary_crossentropy', optimizer='adam', metrics= ["accuracy"])

clf.fit(X_train, y_train, epochs=1000,batch_size=512, validation_split=0.1)

"""En Iyi Sonuc"""

clf = Sequential()
clf.add(Dense(units= 64, activation= "relu", input_dim= 5))
clf.add(Dropout(0.5))
clf.add(Dense(units= 64, activation= "relu"))
clf.add(Dropout(0.5))
clf.add(Dense(units= 651, activation= "sigmoid"))

clf.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics= ["mae"])

clf.fit(X_train, y_train, epochs=1000,batch_size=512, validation_split=0.1)

"""Tahmin"""

y_pred = clf.predict(X_test)

print(y_pred)

"""Sonuc"""

clf.evaluate(X_test, y_test)

print("Kullaninal Olcumler = ",clf.metrics_names)

clf.optimizer

clf.get_layer

clf.get_compile_config()

clf.get_config()

clf.get_metrics_result()

clf.summary()

"""Egitilmis Veri Kullanmak icin"""

pickle.dump(clf, open("ev_fitted", "wb"))

ogrenilmis = pickle.load(open("/content/ev_fitted", "rb"))
ogrenilmis.predict(X_test)