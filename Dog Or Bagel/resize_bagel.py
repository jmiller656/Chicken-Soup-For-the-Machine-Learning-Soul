import cv2
for i in range(63):
	filename = "bagel_"+str(i)+".png";
	dog = cv2.imread("Bagels/"+filename);
	resized_dog = cv2.resize(dog, (32, 32));
	cv2.imwrite("Bagels/"+filename,resized_dog);
print "Done!";