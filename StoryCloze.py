import csv
from sklearn.dummy import DummyClassifier
import numpy as np

x_train, y_train, x_test, y_test = [],[],[],[]

with open('data/Training.csv', 'rb') as csvfile:
     csvData = csv.reader(csvfile, delimiter=',', quotechar='|')
     for i, row in enumerate(csvData):
         if (i!=0):
            sentence = ''
            sentence += row [2] + row[3] + row[4] + row[5] + row[6]
            x_train.append(sentence)
            y_train.append(1)


with open('data/Validation.csv', 'rb') as csvfile:
     csvData = csv.reader(csvfile, delimiter=',', quotechar='|')
     for i, row in enumerate(csvData):
         if (i!=0):
            sentence = ''
            sentence += row[1] + row [2] + row[3] + row[4] 
            x_test.append(sentence + row[5])
            x_test.append(sentence + row[6])
            if(row[7] == "1"):
                y_test.append(1)
                y_test.append(0)
                # sentence += row[5]
            else:
                y_test.append(0)
                y_test.append(1)
                # sentence += row[6]
            print "\n"
            print (sentence)

# print(len(y_test))
dummy_classifier = DummyClassifier(strategy="most_frequent")
dummy_classifier.fit( np.reshape(x_train,(-1,1)),y_train )
print(dummy_classifier.score(np.reshape(x_test,(-1,1)), y_test))