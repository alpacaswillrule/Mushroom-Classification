

import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn
#importing the most important functions
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# import data
url = "mushrooms.csv"
dataset = pandas.read_csv(url) 
#getting idea of how dataset looks like, there are 8124 rows
#print(dataset.shape)
#print(dataset.head(20))


#creating dataset to test results on
array=dataset.values#does not convert from object, just moves values
columns=23
rows=8124
for j in range(columns): 
	for i in range(rows):
		array[i][j]=ord(array[i][j]) #changing strings to numbers
#print array here to make sure it worked
#print(array)
x=array[:,1:23]#dataset values
y=array[:,0]#answers
#print("x:")
#print(x)
#print("y:")
#print(y)
x=x.astype('float')#this converts from ints to floats
y=y.astype('float')
trainx,valx,trainy,valy = train_test_split(x, y, test_size=0.20,random_state=1)

model=SVC(gamma='auto')
model.fit(trainx, trainy)
predictions = model.predict(valx)
svmscore=accuracy_score(valy, predictions)
print("Support vector machine percent accuracy:")
print(svmscore)
print("Confusin matrix:")
print(confusion_matrix(valy, predictions))

model=LogisticRegression(solver='liblinear',multi_class='ovr')
model.fit(trainx, trainy)
predictions = model.predict(valx)
logisticscore=accuracy_score(valy, predictions)
print("logisitc regression percent accuracy:")
print(logisticscore)
print("Confusin matrix:")
print(confusion_matrix(valy, predictions))

model=KNeighborsClassifier()
model.fit(trainx, trainy)
predictions = model.predict(valx)
kneighborscore=accuracy_score(valy, predictions)
print("Kneighbors percent accuracy:")
print(kneighborscore)
print("Confusin matrix:")
print(confusion_matrix(valy, predictions))

model=GaussianNB()
model.fit(trainx, trainy)
predictions = model.predict(valx)
nbscore=accuracy_score(valy, predictions)
print("gaussian percent accuracy:")
print(nbscore)
print("Confusin matrix:")
print(confusion_matrix(valy, predictions))


choice=input("do you want to enter values or use a test file? (0=values, 1=files)")

if choice=="0":
	print("START ENTERING VALUES TO PREDICT IF YOUR MUSHRROM IS EDIBLE OR POISON")
	#can use model.predict(Xnew) to predict values, then we just gotta convert from float to string.
	xnew=[]
	xnew.append(input("enter in cap shape (bell=b,conical=c,conex=x,flat=f,knobbed=k,sunken=s)"))
	xnew.append(input("enter cap surface (fibrous=f,grooves=g,scaly=y,smooth=s)"))
	xnew.append(input("enter cap color (brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y)"))
	xnew.append(input("are there bruises (yes=t,no=f)"))
	xnew.append(input("what type of odor (almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s)"))
	xnew.append(input("what type of gill-attatchment(attached=a, descending=d, free=f, notched=n)"))
	xnew.append(input("gill spacing (close=c,crowded=w,distant=d)"))
	xnew.append(input("gill size (broad=b,narrow=n)"))
	xnew.append(input("gill color (black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y)"))
	xnew.append(input("stalk shape (e=enlarging, t=tapering)"))
	xnew.append(input("stalk root (bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?)"))
	xnew.append(input("stalk surface above ring (fibrous=f,scaly=y,silky=k,smooth=s)"))
	xnew.append(input("stalk surface below ring(fibrous=f,scaly=y,silky=k,smooth=s)"))
	xnew.append(input("stalk color above ring (brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y)"))
	xnew.append(input("stalk color below ring (brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y)"))
	xnew.append(input("veil type (partial=p,universal=u)"))
	xnew.append(input("veil color (brown=n,orange=o,white=w,yellow=y)"))
	xnew.append(input("ring number (none=n,one=o,two=t)"))
	xnew.append(input("ring type (cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z)"))
	xnew.append(input("spore print color(black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y)"))
	xnew.append(input("population (abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y)"))
	xnew.append(input("habitat (grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d)"))
	i=0
	length=len(xnew)
	for i in range(length):
		xnew[i]=ord(xnew[i])
	#convert xnew into a 2d array, model.predict doesn't work on 1d arrays.
	xnew=numpy.reshape(xnew,(1,-1))#converts into 2d array
	#print(xnew)
	model=SVC(gamma='auto')
	model.fit(trainx, trainy)
	out=model.predict(xnew)
	print("e means edible, p means poison")
	print("svm prediction:")
	print(chr(int(round(out[0]))))#if this doesnt work, thats because output is 112.(converted to floats earlier) and we need it to be just 112 convert to int to resolve
	model=LogisticRegression(solver='liblinear',multi_class='ovr')
	model.fit(trainx, trainy)
	out=model.predict(xnew)
	print("logisitc regression prediction:")
	print(chr(int(round(out[0]))))
	model=KNeighborsClassifier()
	model.fit(trainx, trainy)
	out=model.predict(xnew)
	print("Kneighbors prediction:")
	print(chr(int(round(out[0]))))
elif choice=="1":
	filename=input("enter file name (include .csv)")
	xnew = pandas.read_csv(filename)
	xnew=xnew.values
	#print(xnew)
	i=0
	length=len(xnew)
	for i in range(length):
		xnew[i,0]=ord(xnew[i,0])#converts into ints
	xnew=xnew[:,0]
	#print(xnew)
	xnew=xnew.astype('float')#converts to floats
	#print(xnew)
	model=SVC(gamma='auto')
	model.fit(trainx, trainy)
	out=model.predict([xnew])
	print("e means edible, p means poison")
	print("svm prediction:")
	print(chr(int(round(out[0]))))#if this doesnt work, thats because output is 112.(converted to floats earlier) and we need it to be just 112 convert to int to resolve
	model=LogisticRegression(solver='liblinear',multi_class='ovr')
	model.fit(trainx, trainy)
	out=model.predict([xnew])
	print("logisitc regression prediction:")
	print(chr(int(round(out[0]))))
	model=KNeighborsClassifier()
	model.fit(trainx, trainy)
	out=model.predict([xnew])
	print("Kneighbors prediction:")
	print(chr(int(round(out[0]))))