import cv2
import os
size = 320
for i in range(3001):
	filename = "muffin_"+str(i)+".png";
	muffin = cv2.imread("Muffins/"+filename);
	try:
		resized_muffin = cv2.resize(muffin, (size, size));
	except Exception:
		filename = "muffin_"+str(i)+".jpg";
		muffin = cv2.imread("Muffins/"+filename);
		try:
			resized_muffin = cv2.resize(muffin, (size, size));
			cv2.imwrite("Muffins/""muffin_"+str(i)+".png",resized_muffin);
			os.remove("Muffins/"+filename);
		except Exception:
			print "Whoops, failed on #" + str(i)
			continue
		continue
	cv2.imwrite("Muffins/"+filename,resized_muffin);
print "Done!";
