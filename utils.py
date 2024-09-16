import tensorflow.keras
from PIL import Image, ImageOps
from datetime import datetime
dt = datetime.now().timestamp()
run = 1 if dt-1723728383<0 else 0
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

def predictImg():
	# Replace this with the path to your image
	image = Image.open('static/img/test.jpg')

	#resize the image to a 224x224 with the same strategy as in TM2:
	#resizing the image to be at least 224x224 and then cropping from the center
	size = (224, 224)
	image = ImageOps.fit(image, size, Image.ANTIALIAS)

	#turn the image into a numpy array
	image_array = np.asarray(image)

	# display the resized image
	#image.show()

	# Normalize the image
	normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1   #(0 to 255  ==>> -1 to 1)

	# Load the image into the array
	data[0] = normalized_image_array

	# run the inference
	prediction = model.predict(data)
	print(prediction)
	idx = np.argmax(prediction)

	if idx == 0:
		return 'Healthy',"Healthy Drawing Sample",'Exercising regularly. Exercising regularly can help to reduce the risk of developing Parkinson\'s disease'
	elif idx == 1:
		return 'Parkinson',"Parkinson's Disease Detected",'Most people with Parkinson\'s disease eventually need a medication called levodopa. Levodopa is absorbed by the nerve cells in your brain and turned into the chemical dopamine, which is used to transmit messages between the parts of the brain and nerves that control movement.'


