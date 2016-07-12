import cv2
import os
for i in range(3001):
	filename = "dog_"+str(i)+".png";
	dog = cv2.imread("Dogs/"+filename);
	try:
		resized_dog = cv2.resize(dog, (32, 32));
	except Exception:
		filename = "dog_"+str(i)+".jpg";
		dog = cv2.imread("Dogs/"+filename);
		try:
			resized_dog = cv2.resize(dog, (32, 32));
			cv2.imwrite("Dogs/""dog_"+str(i)+".png",resized_dog);
			os.remove("Dogs/"+filename);
		except Exception:
			print "Whoops, failed on #" + str(i)
			continue
		continue
	cv2.imwrite("Dogs/"+filename,resized_dog);
print "Done!";
