import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt

def reduce_dim(dataset):
    nsamples, nx, ny = dataset.shape
    return dataset.reshape((nsamples, nx * ny))

x1 = np.load('data\\training\class_a.npy')
x2 = np.load('data\\training\class_b.npy')
field = np.load('data\classification\\field.npy')
x  = np.concatenate((x1, x2), axis=0)

y1 = np.zeros(len(x1))
y2 = np.ones(len(x2))

y = np.concatenate((y1, y2), axis=0)

#print(np.shape(x1), np.shape(x2), np.shape(x))
#print(np.shape(y1), np.shape(y2), np.shape(y))

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2, shuffle=True)

#print(np.shape(x_train), np.shape(y_train), np.shape(x_test), np.shape(y_test))

d2_train_dataset = reduce_dim(x_train)
d2_test_dataset = reduce_dim(x_test)

clf = svm.SVC(gamma = 0.0005, C=10)
clf.fit(d2_train_dataset, y_train)

prediction = clf.predict(d2_test_dataset)

accuracy = 0
for i in range(len(prediction)):
    if prediction[i] == y_test[i]:
        accuracy += 1
print('Accuracy: ', accuracy/len(prediction))

d2_field_dataset = reduce_dim(field)
prediction = clf.predict(d2_field_dataset)
#print(prediction.shape)
labels_field=[1, 0, 0, 0, 0, 1, 0, 0,
                  0, 0, 1, 1, 1, 1, 0, 0,
                  1, 0, 1, 1, 0, 1, 0, 0,
                  1, 1, 0, 1, 1, 1, 1, 0,
                  1, 1, 0, 0, 0, 1, 1, 1,
                  1, 1, 1, 0, 1, 1, 0, 0,
                  1, 1, 0, 1, 1, 0, 1, 1,
                  0, 0, 0, 0, 0, 0, 1, 0,
                  1, 0, 1, 0, 0, 1, 0, 0,
                  0, 1, 1, 0, 0, 0, 0, 0,
                  0, 0, 0, 1, 1, 0, 1, 1,
                  1, 1, 0, 1, 1, 0, 1, 1,
                  1, 1, 0, 1, 0, 0, 1, 1,
                  0, 0, 1, 1, 0, 1, 0, 1,
                  1, 0, 0, 0, 0, 1, 0, 1,
                  0, 1, 1, 0, 0, 0, 0, 1,
                  1, 0, 1, 0, 1, 0, 1, 0,
                  1, 1, 1, 0, 0, 1, 0, 0,
                  1, 1, 1, 0, 1, 0, 0, 0,
                  1, 1, 1, 1, 0, 1, 0, 1,
                  0, 1, 0, 0, 0, 0, 1, 0,
                  0, 0, 1, 1, 0, 0, 0, 0,
                  1, 1, 1, 1, 0, 1, 0, 1,
                  1, 1, 1, 0, 0, 0, 1, 1,
                  1, 0, 0, 0, 0, 1, 1, 0]
accuracy_field = 0
for i in range(0,200):
    if prediction[i] == labels_field[i]:
        accuracy_field += 1
    #plt.imshow(field[i], interpolation='nearest')
    #plt.savefig('images\\field'+ str(i) +'.png')
print(accuracy_field/len(labels_field))
