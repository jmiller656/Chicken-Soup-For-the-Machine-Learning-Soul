import cv2
import os
size = 320
for i in range(3001):
	filename = "towel_"+str(i)+".png";
	towel = cv2.imread("Towels/"+filename);
	try:
		resized_towel = cv2.resize(towel, (size, size));
	except Exception:
		filename = "towel_"+str(i)+".jpg";
		towel = cv2.imread("Towels/"+filename);
		try:
			resized_towel = cv2.resize(towel, (size, size));
			cv2.imwrite("Towels/""towel_"+str(i)+".png",resized_towel);
			os.remove("Towels/"+filename);
		except Exception:
			print "Whoops, failed on #" + str(i)
			continue
		continue
	cv2.imwrite("Towels/"+filename,resized_towel);
print "Done!";
