import numpy as np
from ImageClassificationConvNet import ImageClassificationConvNet
import torch
from sklearn import metrics

MODEL_STORE_PATH = 'C:\\Users\Ewa\Documents\PycharmProjects\\binaryimageclassification\pytorch_models\\'

def load_data(path):
    dataset = np.load(path)
    dataset = np.expand_dims(dataset, axis=1)
    dataset = torch.from_numpy(dataset)
    return dataset.float()

dataset = load_data('data/classification/field.npy')

#labels eyeballed by a single person. May contain human error, please check
labels_field = [1, 0, 0, 0, 0, 1, 0, 0,
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

classes = ('A', 'B')

cnn = ImageClassificationConvNet()
cnn.eval()
cnn.load_state_dict(torch.load(MODEL_STORE_PATH + 'conv_net_model.ckpt'))
outputs = cnn(dataset)
_, predicted = torch.max(outputs, 1)

#print(predicted)
# calculate accuracy
print('Classification accuracy is {:.{prec}f} %'.format(metrics.accuracy_score(predicted, labels_field) * 100, prec = 2))

# Calculate null accuracy by examining the class distribution of the validation set
predicted = predicted.numpy()
print('A simple classifier accuracy is: {:.{prec}f} %'.format(max(predicted.mean(), 1 - predicted.mean()) * 100, prec = 2))

# Check confusion matrix
confusion = metrics.confusion_matrix(labels_field, predicted)
print("Confusion matrix: \n", confusion)

# Classification error
classification_error = 1 - metrics.accuracy_score(predicted, labels_field)
print('Classification error is: {:.{prec}f} %'.format(classification_error * 100, prec = 2))

#Sensitivity/recall, i.e. How sensitive is the classifier to detecting positive instances
sensitivity = metrics.recall_score(predicted, labels_field)
print('True positive rate is: {:.{prec}f} %'.format(sensitivity * 100, prec = 2))

#Specificity, i.e. When the actual value is negative, how often the prediction is correct
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]
specificity = TN / (TN + FP)
print('Specificity: {:.{prec}f} %'.format(specificity * 100, prec =2))

#False Positive Rate, i.e. When the actual value is negative, how often the prediction is incorrect
false_positive_rate = FP / float(TN + FP)
print('False positive rate: {:.{prec}f} %'.format(false_positive_rate * 100, prec =0))

# Precision, i.e. When a positive value is predicted, how often the prediction is correct
precision = TP / float(TP + FP)
print('Precision rate: {:.{prec}f} %'.format(precision * 100, prec =2))

#print('Predicted: ', ' '.join('%2s' % classes[predicted[j]] for j in range(len(predicted))))