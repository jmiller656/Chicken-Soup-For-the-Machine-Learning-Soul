import cv2
for i in range(63):
	filename = "dog_"+str(i)+".png";
	dog = cv2.imread("Dogs/"+filename);
	resized_dog = cv2.resize(dog, (32, 32));
	cv2.imwrite("Dogs/"+filename,resized_dog);
print "Done!";