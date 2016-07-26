import cv2
import os
size = 320
for i in range(3001):
	filename = "fc_"+str(i)+".png";
	fc = cv2.imread("Fried Chickens/"+filename);
	try:
		resized_fc = cv2.resize(fc, (size, size));
	except Exception:
		filename = "fc_"+str(i)+".jpg";
		fc = cv2.imread("Fried Chickens/"+filename);
		try:
			resized_fc = cv2.resize(fc, (size, size));
			cv2.imwrite("Fried Chickens/""fc_"+str(i)+".png",resized_fc);
			os.remove("Fried Chickens/"+filename);
		except Exception:
			print "Whoops, failed on #" + str(i)
			continue
		continue
	cv2.imwrite("Fried Chickens/"+filename,resized_fc);
print "Done!";
