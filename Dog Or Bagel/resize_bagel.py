import cv2
import os
for i in range(3001):
	filename = "bagel_"+str(i)+".png";
	bagel = cv2.imread("Bagels/"+filename);
	try:
		resized_bagel = cv2.resize(bagel, (32, 32));
	except Exception:
		filename = "bagel_"+str(i)+".jpg";
		bagel = cv2.imread("Bagels/"+filename);
		try:
			resized_bagel = cv2.resize(bagel, (32, 32));
			cv2.imwrite("Bagels/""bagel_"+str(i)+".png",resized_bagel);
			os.remove("Bagels/"+filename);
		except Exception:
			print "Whoops, failed on #" + str(i)
			continue
		continue
	cv2.imwrite("Bagels/"+filename,resized_bagel);
print "Done!";
