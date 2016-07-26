import cv2
import os
size = 320
for i in range(3001):
	filename = "croissant_"+str(i)+".png";
	croissant = cv2.imread("Croissants/"+filename);
	try:
		resized_croissant = cv2.resize(croissant, (size, size));
	except Exception:
		filename = "croissant_"+str(i)+".jpg";
		croissant = cv2.imread("Croissants/"+filename);
		try:
			resized_croissant = cv2.resize(croissant, (size, size));
			cv2.imwrite("Croissants/""croissant_"+str(i)+".png",resized_croissant);
			os.remove("Croissants/"+filename);
		except Exception:
			print "Whoops, failed on #" + str(i)
			continue
		continue
	cv2.imwrite("Croissants/"+filename,resized_croissant);
print "Done!";
