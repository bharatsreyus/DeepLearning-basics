import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np

#load the data
mnist = tf.keras.datasets.mnist

#split the data
(x_train,y_train),(x_test,y_test) = mnist.load_data()

#plot 
#plt.imshow(x_train[1], cmap = plt.cm.binary)

#normalize the data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#print(x_train[1])

#build the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))

#train the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=3)

#evaluate the model
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss,val_acc)

#save model
model.save('num_reader.model')

#load model
new_model = tf.keras.models.load_model('num_reader.model')

#predict
predictions = new_model.predict(x_test)

#to show the image of the number
plt.imshow(x_test[1])

#obtains the number from probability ditribution list using numpy
print(np.argmax(predictions[1]))
