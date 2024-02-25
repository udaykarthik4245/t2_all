from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D,Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
import numpy as np

. Train this network using a categorical cross-entropy loss
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32') / 255
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255

# Split the training data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
#                                model
model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))  #make change 32 to 64 for e bit
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))                         # make change 128 to 256
model.add(Dense(10,activation='softmax'))
                    #using adam optimizer
# optimizer = Adam(learning_rate=0.001,beta_1=0.99,beta_2=0.999)
# model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

                    #using stochastic gradient descent optimzer
optimizer = SGD(learning_rate=0.001, momentum=0.99)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

                                #model training
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
model_checkpoint = ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)

# Training the model with callbacks
history = model.fit(x_train, y_train,
                    epochs=5,
                    batch_size=32,
                    validation_split=10,
                    validation_data=(x_val, y_val),
                    callbacks=[early_stopping, model_checkpoint])

# Load the best model saved by ModelCheckpoint
model.load_weights('best_model.h5')

# Evaluate the model for test sets
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

# Print training loss, training error, and validation error
for epoch in range(len(history.history['loss'])):
    print(f'Epoch {epoch+1}:')
    print(f'Training Loss: {history.history["loss"][epoch]}')
    print(f'Training Error: {1 - history.history["accuracy"][epoch]}')
    print(f'Validation Error: {1 - history.history["val_accuracy"][epoch]}')
    print('------------------------')

print('acc',test_acc*100)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
# history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_val, y_val))

# # Evaluate the model on the test set
# test_loss, test_acc = model.evaluate(x_test, y_test)
# print('Test accuracy:', test_acc)
# print('accuracy:',history.history['accuracy'])






# Test accuracy: 0.9872999787330627
# Epoch 1:
# Training Loss: 0.3178206980228424
# Training Error: 0.09642595052719116
# Validation Error: 0.023333311080932617
# ------------------------
# Epoch 2:
# Training Loss: 0.06026061251759529
# Training Error: 0.018185198307037354
# Validation Error: 0.01616668701171875
# ------------------------
# Epoch 3:
# Training Loss: 0.03997709974646568
# Training Error: 0.012203693389892578
# Validation Error: 0.013166666030883789
# ------------------------
# Epoch 4:
# Training Loss: 0.03145258128643036
# Training Error: 0.009962975978851318
# Validation Error: 0.012499988079071045
# ------------------------
# Epoch 5:
# Training Loss: 0.02547294832766056
# Training Error: 0.008129656314849854
# Validation Error: 0.012333333492279053
# ------------------------
# acc 98.72999787330627


#after increasing the network size from convo2d 32 to 64 and dense 128 to 256
#
# Test accuracy: 0.9907000064849854
# Epoch 1:
# Training Loss: 0.3136420249938965
# Training Error: 0.0911296010017395
# Validation Error: 0.02266669273376465
# ------------------------
# Epoch 2:
# Training Loss: 0.051164936274290085
# Training Error: 0.015777766704559326
# Validation Error: 0.01583331823348999
# ------------------------
# Epoch 3:
# Training Loss: 0.03523305431008339
# Training Error: 0.01066666841506958
# Validation Error: 0.011666655540466309
# ------------------------
# Epoch 4:
# Training Loss: 0.027716070413589478
# Training Error: 0.009055554866790771
# Validation Error: 0.01466667652130127
# ------------------------
# Epoch 5:
# Training Loss: 0.02124771662056446
# Training Error: 0.006537020206451416
# Validation Error: 0.01066666841506958
# ------------------------
# acc 99.07000064849854
