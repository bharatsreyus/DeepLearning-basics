import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle

#pickle in features and labels
pickle_in = open('X_catvsdog','rb')
X = pickle.load(pickle_in)
pickle_in = open('y_catvsdog','rb')
y = pickle.load(pickle_in)

X = X/255.0 #set the values in between 0 and 1

model = Sequential()

model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]) )
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(X, y, batch_size = 32, epochs=10, validation_split= 0.1)
                                                                       
model.save('catvsdog_classifier.model')
new_model = tf.keras.models.load_model('catvsdog_classifier.model')   
