import cv2
import os
size = 320
for i in range(3001):
	filename = "bread_"+str(i)+".png";
	bread = cv2.imread("Breads/"+filename);
	try:
		resized_bread = cv2.resize(bread, (size, size));
	except Exception:
		filename = "bread_"+str(i)+".jpg";
		bread = cv2.imread("Breads/"+filename);
		try:
			resized_bread = cv2.resize(bread, (size, size));
			cv2.imwrite("Breads/""bread_"+str(i)+".png",resized_bread);
			os.remove("Breads/"+filename);
		except Exception:
			print "Whoops, failed on #" + str(i)
			continue
		continue
	try:
		os.remove("Breads/bread_"+str(i)+".jpg");
	except Exception:
		continue
	cv2.imwrite("Breads/"+filename,resized_bread);
	
print "Done!";
