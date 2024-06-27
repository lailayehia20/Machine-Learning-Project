import samples
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

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
#digitdata
train_img, train_labels = getData(5000,"digitdata","training",28,28)
val_img, val_labels = getData(1000, "digitdata","validation",28,28)
test_img, test_labels = getData(1000, "digitdata","test",28,28)

#facedata
ftrain_img, ftrain_labels = getData(451,"facedata","facedatatrain",60,70)
fval_img, fval_labels = getData(301, "facedata","facedatavalidation",60,70)
ftest_img, ftest_labels = getData(150, "facedata","facedatatest",60,70)


max1 = 0
max2 = 0
a=[]
b=[]

#calling MLPClassifier function and changing the hyperparameters
#case 1:
digital_clf0 = MLPClassifier(hidden_layer_sizes=3, activation='identity', learning_rate='constant', learning_rate_init=0.01 )
face_clf0 = MLPClassifier(hidden_layer_sizes=3, activation='identity', learning_rate='constant', learning_rate_init=0.01)
#case 2:
digital_clf1 =MLPClassifier(hidden_layer_sizes=8, activation='logistic', learning_rate='adaptive', learning_rate_init=0.2)
face_clf1 =MLPClassifier(hidden_layer_sizes=8, activation='logistic', learning_rate='adaptive', learning_rate_init=0.2)
#case 3:
digital_clf2 =MLPClassifier()
face_clf2 =MLPClassifier()
#case 4:
digital_clf3 =MLPClassifier(hidden_layer_sizes=20, activation='tanh', learning_rate='adaptive', learning_rate_init=0.2)
face_clf3 =MLPClassifier(hidden_layer_sizes=20, activation='tanh', learning_rate='adaptive', learning_rate_init=0.2)
#case 5:
digital_clf4 =MLPClassifier(hidden_layer_sizes=4, activation='relu', learning_rate='constant', learning_rate_init=0.04)
face_clf4 =MLPClassifier(hidden_layer_sizes=4, activation='relu', learning_rate='constant', learning_rate_init=0.04)
#case 6:
digital_clf5 =MLPClassifier(hidden_layer_sizes=50, activation='logistic', learning_rate='constant', learning_rate_init=0.6)
face_clf5 =MLPClassifier(hidden_layer_sizes=50, activation='logistic', learning_rate='constant', learning_rate_init=0.6)
#case 7:
digital_clf6 =MLPClassifier(hidden_layer_sizes=1, activation='identity', learning_rate='constant', learning_rate_init=0.036)
face_clf6 =MLPClassifier(hidden_layer_sizes=1, activation='identity', learning_rate='constant', learning_rate_init=0.036)

#fitting data
digital_clf0.fit(train_img, train_labels)
face_clf0.fit(ftrain_img, ftrain_labels)
digital_clf1.fit(train_img, train_labels)
face_clf1.fit(ftrain_img, ftrain_labels)
digital_clf2.fit(train_img, train_labels)
face_clf2.fit(ftrain_img, ftrain_labels)
digital_clf3.fit(train_img, train_labels)
face_clf3.fit(ftrain_img, ftrain_labels)
digital_clf4.fit(train_img, train_labels)
face_clf4.fit(ftrain_img, ftrain_labels)
digital_clf5.fit(train_img, train_labels)
face_clf5.fit(ftrain_img, ftrain_labels)
digital_clf6.fit(train_img, train_labels)
face_clf6.fit(ftrain_img, ftrain_labels)




#validation images
Digital_Acc1 = accuracy(digital_clf0, val_img, val_labels)
a.append(Digital_Acc1)
Face_Acc1 = accuracy(face_clf0, fval_img, fval_labels)
b.append(Face_Acc1)
Digital_Acc2 = accuracy(digital_clf1, val_img, val_labels)
a.append(Digital_Acc2)
Face_Acc2 = accuracy(face_clf1, fval_img, fval_labels)
b.append(Face_Acc2)
Digital_Acc3 = accuracy(digital_clf2, val_img, val_labels)
a.append(Digital_Acc3)
Face_Acc3 = accuracy(face_clf2, fval_img, fval_labels)
b.append(Face_Acc3)
Digital_Acc4 = accuracy(digital_clf3, val_img, val_labels)
a.append(Digital_Acc4)
Face_Acc4 = accuracy(face_clf3, fval_img, fval_labels)
b.append(Face_Acc4)
Digital_Acc5 = accuracy(digital_clf4, val_img, val_labels)
a.append(Digital_Acc5)
Face_Acc5 = accuracy(face_clf4, fval_img, fval_labels)
b.append(Face_Acc5)
Digital_Acc6 = accuracy(digital_clf5, val_img, val_labels)
a.append(Digital_Acc6)
Face_Acc6 = accuracy(face_clf5, fval_img, fval_labels)
b.append(Face_Acc6)
Digital_Acc7 = accuracy(digital_clf6, val_img, val_labels)
a.append(Digital_Acc7)
Face_Acc7 = accuracy(face_clf6, fval_img, fval_labels)
b.append(Face_Acc7)


#printing the accuracies of all the cases
print("list of digital data accuracies in each case: ")
print(a)
print("list of Face data accuracies in each case: ")
print(b)

index1 = 0
index2 = 0
for i in range(len(a)):
    if a[i]>max1:
        max1=a[i]
        index1=i
for j in range(len(b)):
    if b[j]>max2:
        max2=b[j]
        index2=j

D_clfs_list=[digital_clf0,digital_clf1,digital_clf2,digital_clf3,digital_clf4,digital_clf5,digital_clf6]
f_clfs_list=[face_clf0,face_clf1,face_clf2,face_clf3,face_clf4,face_clf5,face_clf6]

print("maximum accuracy on digital validation images in case of digital data is "+ str(a[index1]))
print("maximum accuracy on face validation images in case of digital data is "+ str(b[index2]))

#testing images
print("In case of digitdata: ")
print("The Accuracy obtained when testing on test images " + str(accuracy(D_clfs_list[index1], test_img, test_labels)))

print("In case of facedata: ")
print("The Accuracy obtained when testing on test images " + str(accuracy(f_clfs_list[index2], ftest_img, ftest_labels)))
