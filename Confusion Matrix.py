TP = 3
FP = 1
TN = 1
FN = 2


def errorRate():
    error = (FP + FN)/(TP + TN + FP + FN)
    print('Error Rate: {}'.format(error))


def accuracy():
    acc = (TP + TN)/(TP + TN + FP + FN)
    print('Accuracy: {}'.format(acc))


def recall():
    recal = (TP)/(TP + FN)
    print('Recall: {}'.format(recal))


def precision():
    precise = (TP)/(TP + FP)
    print('Precision: {}'.format(precise))


def f1score():
    f1 = (2 * TP)/(2 * TP + FP + FN)
    print('F1 Score: {}'.format(f1))


errorRate()
accuracy()
recall()
precision()
f1score()