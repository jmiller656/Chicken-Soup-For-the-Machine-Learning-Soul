import cv2
import os
size = 320
for i in range(3001):
	filename = "dog_"+str(i)+".png";
	dog = cv2.imread("Dogs/"+filename);
	try:
		resized_dog = cv2.resize(dog, (size, size));
	except Exception:
		filename = "dog_"+str(i)+".jpg";
		dog = cv2.imread("Dogs/"+filename);
		try:
			resized_dog = cv2.resize(dog, (size, size));
			cv2.imwrite("Dogs/""dog_"+str(i)+".png",resized_dog);
			os.remove("Dogs/"+filename);
		except Exception:
			print "Whoops, failed on #" + str(i)
			continue
		continue
	try:
		os.remove("Dogs/dog_"+str(i)+".jpg");
	except Exception:
		continue
	cv2.imwrite("Dogs/"+filename,resized_dog);
	
print "Done!";
