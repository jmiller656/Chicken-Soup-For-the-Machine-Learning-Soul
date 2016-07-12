import cv2
import numpy as np
#Method used to convert type of image from uint8 -> float32 for ML purposes
def convertImage(image):
	#return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32,copy=False).T.flatten();
	return image.astype(np.float32,copy=False).T.flatten();

#Method used to get my (admittedly small) training dataset for dogs and bagels
def get_train_dataset():
	train_data = [];
	train_labels = [];
	dogString = "Dogs/dog_";
	bagelString = "Bagels/bagel_";
	i = 10;
	while(i<3001):
		dog = convertImage(cv2.imread(dogString+ str(i) +".png"));
		bagel = convertImage(cv2.imread(bagelString + str(i) + ".png"));
		train_data.append(dog);
		train_labels.append([1,0]);
		train_data.append(bagel);
		train_labels.append([0,1]);
		i= i +1;
	return [np.array(train_data),np.array(train_labels)];

#Method used to get my (even smaller) testing dataset for dogs and bagels
def get_test_dataset():
	test_data = [];
	test_labels = [];
	dogString = "Dogs/dog_";
	bagelString = "Bagels/bagel_";
	for i in range(10):
		dog = convertImage(cv2.imread(dogString+ str(i) +".png"));
		bagel = convertImage(cv2.imread(bagelString + str(i) + ".png"));
		test_data.append(dog);
		test_labels.append([1,0]);
		test_data.append(bagel);
		test_labels.append([0,1]);
	return [np.array(test_data),np.array(test_labels)];

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
