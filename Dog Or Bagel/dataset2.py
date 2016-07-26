import cv2
import numpy as np
prefixes = [
	"Dogs/dog_",
	"Bagels/bagel_",
	"Croissants/croissant_",
	"Fried Chickens/fc_",
	"Muffins/muffin_",
	"Teddy Bears/tb_",
	"Towels/towel_"
]
classifiers = [
	[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
	[0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0],
	[0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0],
	[0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0],
	[0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0],
	[0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0],
	[0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0],
	[0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
]
#Method used to convert type of image from uint8 -> float32 for ML purposes
def convertImage(image):
	return image.astype(np.float32,copy=False).T.flatten();

#Method used to get my (admittedly small) training dataset for dogs and bagels
def get_train_dataset(size=32,start=10,end=3001):
	train_data = [];
	train_labels = [];
	i = start;
	while(i<end):
		for j in range(len(classifiers)-1):
			img = cv2.imread(prefixes[j]+str(i)+".png")
			try:
				img = cv2.resize(img,(size,size))
			except:
				print "Failed at: " + prefixes[j]+str(i)+".png"		
			train_data.append(convertImage(img))
			train_labels.append(classifiers[j])
		i+=1
	return [np.array(train_data),np.array(train_labels)];

#Method used to get my (even smaller) testing dataset for dogs and bagels
def get_test_dataset(size=32,start=0,end=10):
	test_data = [];
	test_labels = [];
	i = start
	while(i<end):
		for j in range(len(classifiers)-1):
			img = cv2.imread(prefixes[j]+str(i)+".png")
			img = cv2.resize(img,(size,size))
			test_data.append(convertImage(img))
			test_labels.append(classifiers[j])
		i+=1
	return [np.array(test_data),np.array(test_labels)];

