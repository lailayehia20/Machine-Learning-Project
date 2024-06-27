import samples
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt


def getData(num, foldername, filename, x, y):
    img = samples.loadDataFile("data/" + foldername + "/" + filename + "images", num, x, y)
    labels = samples.loadLabelsFile("data/" + foldername + "/" + filename + "labels", num)
    images = []
    for i in range(len(img)):
        images.append(np.array(img[i].getPixels()).flatten())
    return images, labels


def accuracy(knn, vimages, vlabels):
    predictions = knn.predict(vimages)
    count = 0
    for i in range(len(predictions)):
        if predictions[i] == vlabels[i]:
            count += 1
    return count / len(predictions) * 100


# digitdata
train_img, train_labels = getData(5000, "digitdata", "training", 28, 28)
val_img, val_labels = getData(1000, "digitdata", "validation", 28, 28)
test_img, test_labels = getData(1000, "digitdata", "test", 28, 28)

# facedata
ftrain_img, ftrain_labels = getData(451, "facedata", "facedatatrain", 60, 70)
fval_img, fval_labels = getData(301, "facedata", "facedatavalidation", 60, 70)
ftest_img, ftest_labels = getData(150, "facedata", "facedatatest", 60, 70)

maxAccuracy1 = 0
maxAccuracy2 = 0

bestKnn = None
x1 = []
x2 = []
a = []
b = []
c = []
d = []

for i in range(10):
    x1.append(i + 2)
    x2.append(i + 2)
    for j in range(2):
        print("Tyring k = " + str(i + 2) + " With p=" + str(j + 1))
        knn = KNeighborsClassifier(n_neighbors=i + 2, p=j + 1)
        knnf = KNeighborsClassifier(n_neighbors=i + 2, p=j + 1)
        knn.fit(train_img, train_labels)
        knnf.fit(ftrain_img, ftrain_labels)
        currAcc1 = accuracy(knn, val_img, val_labels)
        currAcc2 = accuracy(knnf, fval_img, fval_labels)
        print("curr of digitdata: " + str(currAcc1))
        print("curr of facedata: " + str(currAcc2))

        if j == 0:
            a.append(currAcc1)
            b.append(currAcc2)
        if j == 1:
            c.append(currAcc1)
            d.append(currAcc2)

        if maxAccuracy1 < currAcc1:
            maxAccuracy1 = currAcc1
            maxk = i + 2
            distance = j + 1
            bestKnn = knn

        if maxAccuracy2 < currAcc2:
            maxAccuracy2 = currAcc2
            maxk2 = i + 2
            distance2 = j + 1
            bestKnn2 = knnf

#plotting Digit Data in case of Eculidean distance and Manhattan Distance
plt.xlabel('K values of digit data')
plt.ylabel('Accuracy of given k')
plt.title('Effect of changing k on Digit Data ')
plt.plot(x1,a, color='red',label='DigitData in case of Eculidean')
plt.plot(x1,c, color='blue',label=' DigitData in case of Manhattan')
plt.legend()
plt.show()

#plotting Face Data in case of Eculidean distance and Manhattan Distance
plt.xlabel('K values of digit data')
plt.ylabel('Accuracy of given k')
plt.title('Effect of changing k on Face Data ')
plt.plot(x2,b, color='green',label=' FaceData in case of Eculidean')
plt.plot(x2,d, color='yellow',label=' FaceData in case of Manhattan')
plt.legend()
plt.show()



print("In case of digitdata: ")
print("Max Accuracy is obtained when k = " + str(maxk))
if distance == 1:
    print("The max accuracy is obtained in the case of Euclidean distance")
else:
    print("The max accuracy is obtained in the case of Manhatten distance")

print("The best accuracy " + str(accuracy(bestKnn, test_img, test_labels)))


print("In case of facedata: ")
print("Max Accuracy is obtained when k = " + str(maxk2))
if distance2 == 1:
    print("The max accuracy is obtained in the case of Euclidean distance")
else:
    print("The max accuracy is obtained in the case of Manhatten distance")

print("The best accuracy " + str(accuracy(bestKnn2, ftest_img, ftest_labels)))