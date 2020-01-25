#CATS VS DOGS
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import random
import pickle 

DATADIR = "G:\\repository\\kagglecatsanddogs_3367a\\PetImages" #escseq
CATEGORIES = ["Dog","Cat"]


training_data = []
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category) #0 for dog 1 for cat
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                #plt.imshow(img_array, cmap="gray")
                new_array = cv2.resize(img_array, (50,50)) #50x50 img size
                #plt.imshow(new_array, cmap='gray')
                training_data.append([new_array, class_num]) #creatin trainin data
            except  Exception as e:
                pass
            
create_training_data()

print(len(training_data))

random.shuffle(training_data) #shuffle the data 

X=[] #features
y= [] #labels

for features , labels in training_data:
    X.append(features)
    y.append(labels)

X= np.array(X).reshape(-1, 50,50, 1) #feed our model numpy array

#pickle the datasets
pickle_out = open("X_catvsdog" , "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y_catvsdog" , "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
