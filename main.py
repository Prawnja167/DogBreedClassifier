from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image  
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

img_path = "Sample Images/6.jpg"
ResNet50_model = ResNet50(weights='imagenet')
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
img = preprocess_input(np.expand_dims(x, axis=0))
prediction = np.argmax(ResNet50_model.predict(img))

if prediction >= 151 or prediction <= 268 :
    dog_breed = decode_predictions(ResNet50_model.predict(img))[0][0][1]
    dog_breed = dog_breed.replace("_", " ")
    result = "This dog is most likely a " + dog_breed
else :
    result = "This is not a dog!"

cv2_img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)
plt.imshow(cv2_img)
plt.title(result)
plt.axis("off")
plt.show()
