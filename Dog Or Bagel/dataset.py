import cv2
import numpy as np
#Method used to convert type of image from uint8 -> float32 for ML purposes
def convertImage(image):
	return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32,copy=False).T.flatten();

#Method used to get my (admittedly small) training dataset for dogs and bagels
def get_train_dataset():
	train_data = [];
	dogString = "Dogs/dog_";
	bagelString = "Bagels/bagel_";
	for i in range(5):
		dog = convertImage(cv2.imread(dogString+ str(i) +".png"));
		bagel = convertImage(cv2.imread(bagelString + str(i) + ".png"));
		train_data.append(dog);
		train_data.append(bagel);
	return np.array(train_data);

#Method used to get my (even smaller) testing dataset for dogs and bagels
def get_test_dataset():
	test_data = [];
	dogString = "Dogs/dog_";
	bagelString = "Bagels/bagel_";
	for i in range(3):
		dog = convertImage(cv2.imread(dogString+ str(i+5) +".png"));
		bagel = convertImage(cv2.imread(bagelString + str(i+5) + ".png"));
		test_data.append(dog);
		test_data.append(bagel);
	return np.array(test_data);

#Method to get labels for training set
def get_train_labels():
	labels = [[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1]];
	labels = np.array(labels);
	return labels.astype(np.float32,copy=False);

#Method to get test labels
def get_test_labels():
	labels = [[1,0],[0,1],[1,0],[0,1],[1,0],[0,1]];
	labels = np.array(labels);
	return labels.astype(np.float32,copy=False);