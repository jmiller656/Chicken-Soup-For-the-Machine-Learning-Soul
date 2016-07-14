import cv2
import numpy as np
#Method used to convert type of image from uint8 -> float32 for ML purposes
def convertImage(image):
	#return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32,copy=False).T.flatten();
	return image.astype(np.float32,copy=False).T.flatten();

#Method used to get my (admittedly small) training dataset for dogs and bagels
def get_train_dataset(size=32,start=10,end=3001):
	train_data = [];
	train_labels = [];
	dogString = "Dogs/dog_";
	bagelString = "Bagels/bagel_";
	i = start;
	while(i<end):
		dog = cv2.imread(dogString+ str(i) +".png");
		bagel = cv2.imread(bagelString + str(i) + ".png");
		dog = cv2.resize(dog, (size, size));
		bagel = cv2.resize(bagel, (size, size));
		dog = convertImage(dog);
		bagel = convertImage(bagel);
		train_data.append(dog);
		train_labels.append([1.0,0.0]);
		train_data.append(bagel);
		train_labels.append([0.0,1.0]);
		i= i +1;
	return [np.array(train_data),np.array(train_labels)];

#Method used to get my (even smaller) testing dataset for dogs and bagels
def get_test_dataset(size=32,start=0,end=10):
	test_data = [];
	test_labels = [];
	dogString = "Dogs/dog_";
	bagelString = "Bagels/bagel_";
	i = start
	while(i<end):
		dog = cv2.imread(dogString+ str(i) +".png");
		bagel = cv2.imread(bagelString + str(i) + ".png");
		dog = cv2.resize(dog, (size, size));
		bagel = cv2.resize(bagel, (size, size));
		dog = convertImage(dog);
		bagel = convertImage(bagel);
		test_data.append(dog);
		test_labels.append([1.0,0.0]);
		test_data.append(bagel);
		test_labels.append([0.0,1.0]);
		i += 1;
	return [np.array(test_data),np.array(test_labels)];

