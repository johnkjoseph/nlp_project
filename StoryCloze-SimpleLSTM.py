
from __future__ import print_function

import numpy as np
import keras
import csv
from sklearn.dummy import DummyClassifier
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Activation
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

baseline = 0
dev = 1
max_words = 1000
batch_size = 32
epochs = 400
numberOfinputs = 600

x_train, y_train, x_test, y_test = [],[],[],[]

# with open('data/Training.csv', 'rb') as csvfile:
#      csvData = csv.reader(csvfile)
#      for i, row in enumerate(csvData):
#         if (i!=0):
#             sentence = ''
#             sentence += row [2] + row[3] + row[4] + row[5] + row[6]
#             x_train.append(sentence)
#             y_train.append(1)

#         if (dev and i == 100):
#             break

# with open('data/Training2.tsv', 'rb') as csvfile:
#      csvData = csv.reader(csvfile, delimiter='\t')
#      for i, row in enumerate(csvData):
#         if (i!=0 and i<=400):
#             sentence = ''
#             sentence += row[1] + row [2] + row[3] + row[4] 

#             x_train.append(sentence + row[5])
#             if(row[7] == "1"):
#                 y_train.append(1)
#             else:
#                 y_train.append(0)
#         elif (i!=0):
#             sentence = ''
#             sentence += row[1] + row [2] + row[3] + row[4] 

#             x_test.append(sentence + row[5])
#             if(row[7] == "1"):
#                 y_test.append(1)
#             else:
#                 y_test.append(0)
#         if (dev and i == 500):
#             break



with open('data/Validation.csv', 'rb') as csvfile:
     csvData = csv.reader(csvfile)
     for i, row in enumerate(csvData):
        if (i!=0):
            sentence = ''
            
            sentence += row[1] + row [2] + row[3] + row[4] 
            x_test.append(sentence + row[5])
            if(row[7] == "1"):
                y_test.append(1)
            else:
                y_test.append(0)
        if (dev and i == numberOfinputs):
             break

sliceVal = int(numberOfinputs*.8)

x_train = list(x_test[:sliceVal])
y_train = list(y_test[:sliceVal])

x_test = list(x_test[sliceVal:])
y_test = list(y_test[sliceVal:])

stop_here_please = EarlyStopping(patience=5)

if(baseline):
    dummy_classifier = DummyClassifier(strategy="most_frequent")
    dummy_classifier.fit( np.reshape(x_train,(-1,1)),y_train )
    print(dummy_classifier.score(np.reshape(x_test,(-1,1)), y_test))
    exit()

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

# num_classes = np.max(y_train) + 1
# print(num_classes, 'classes')

print('Vectorizing sequence data...')
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train+x_test)

# print(tokenizer.word_counts)
# print(tokenizer.document_count)
# print(tokenizer.word_index)
# print(tokenizer.word_index['noticed'])
# print(tokenizer.word_docs)
# print(x_train[0])
x_train = tokenizer.texts_to_matrix(x_train, mode='binary')
x_test = tokenizer.texts_to_matrix(x_test, mode='binary')
np.set_printoptions(threshold='nan')
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)


print('Build model...')
model = Sequential()


model.add(Embedding(1202, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test), callbacks=[stop_here_please])
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)

print('Test score:', score)
print('Test accuracy:', acc)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()