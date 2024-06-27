import samples
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

def getData(num,foldername,filename,x,y):
    img = samples.loadDataFile("data/"+foldername+"/"+filename+"images",num,x,y)
    labels=samples.loadLabelsFile("data/"+foldername+"/"+filename+"labels",num)
    images = []
    for i in range(len(img)):
        images.append(np.array(img[i].getPixels()).flatten())
    return images, labels

def accuracy(gnb,vimages,vlabels):
    predictions = gnb.predict(vimages)
    count = 0
    for i in range(len(predictions)):
        if predictions[i] == vlabels[i]:
            count += 1
    return count / len(predictions) * 100
#digitdata
train_img, train_labels = getData(5000,"digitdata","training",28,28)
val_img, val_labels = getData(1000, "digitdata","validation",28,28)
test_img, test_labels = getData(1000, "digitdata","test",28,28)

#facedata
ftrain_img, ftrain_labels = getData(451,"facedata","facedatatrain",60,70)
fval_img, fval_labels = getData(301, "facedata","facedatavalidation",60,70)
ftest_img, ftest_labels = getData(150, "facedata","facedatatest",60,70)


maxAccuracy1 = 0
maxAccuracy2 = 0

gnb = GaussianNB()
gnb1 = GaussianNB()
gnb.fit(train_img, train_labels)
gnb1.fit(ftrain_img, ftrain_labels)

#validation images
currAcc1 = accuracy(gnb,val_img, val_labels)
currAcc2 = accuracy(gnb1,fval_img,fval_labels)
print("curr of digitdata: " + str(currAcc1))
print("curr of facedata: "+ str(currAcc2))

#testing images
print("In case of digitdata: ")
print("The Accuracy obtained when testing on test images " + str(accuracy(gnb, test_img, test_labels)))

print("In case of facedata: ")
print("The Accuracy obtained when testing on test images " + str(accuracy(gnb1, ftest_img, ftest_labels)))

