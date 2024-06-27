from sklearn import tree
import samples
import numpy as np
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt


def getData(num,foldername,filename,x,y):
    img = samples.loadDataFile("data/"+foldername+"/"+filename+"images",num,x,y)
    labels=samples.loadLabelsFile("data/"+foldername+"/"+filename+"labels",num)
    images = []
    for i in range(len(img)):
        images.append(np.array(img[i].getPixels()).flatten())
    return images, labels

def accuracy(clf,vimages,vlabels):
    predictions = clf.predict(vimages)
    count = 0
    for i in range(len(predictions)):
        if predictions[i] == vlabels[i]:
            count += 1
    return count / len(predictions) * 100

def ploting_tree(clf,vimages,vlabels):
     iris = load_iris()
     vimages, vlabels = iris.data, iris.target
     tree.plot_tree(clf)



#digitdata
train_img, train_labels = getData(5000,"digitdata","training",28,28)
val_img, val_labels = getData(1000, "digitdata","validation",28,28)
test_img, test_labels = getData(1000, "digitdata","test",28,28)

#facedata
ftrain_img, ftrain_labels = getData(451,"facedata","facedatatrain",60,70)
fval_img, fval_labels = getData(301, "facedata","facedatavalidation",60,70)
ftest_img, ftest_labels = getData(150, "facedata","facedatatest",60,70)


clf = tree.DecisionTreeClassifier()
clf1 = tree.DecisionTreeClassifier()
clf.fit(train_img, train_labels)
clf1.fit(ftrain_img, ftrain_labels)


#plotting training data
ploting_tree(clf,val_img, val_labels)
plt.show()
ploting_tree(clf1,fval_img,fval_labels)
plt.show()

#validation images
currAcc1 = accuracy(clf,val_img, val_labels)
currAcc2 = accuracy(clf1,fval_img,fval_labels)
print("accuracy of digitdata: " + str(currAcc1))
print("accuracy of facedata: "+ str(currAcc2))

#testing images
print("In case of digitdata: ")
print("The Accuracy obtained when testing on test images " + str(accuracy(clf, test_img, test_labels)))

print("In case of facedata: ")
print("The Accuracy obtained when testing on test images " + str(accuracy(clf1, ftest_img, ftest_labels)))








