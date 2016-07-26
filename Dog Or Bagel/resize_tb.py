import cv2
import os
size = 320
for i in range(3001):
	filename = "tb_"+str(i)+".png";
	tb = cv2.imread("Teddy Bears/"+filename);
	try:
		resized_tb = cv2.resize(tb, (size, size));
	except Exception:
		filename = "tb_"+str(i)+".jpg";
		tb = cv2.imread("Teddy Bears/"+filename);
		try:
			resized_tb = cv2.resize(tb, (size, size));
			cv2.imwrite("Teddy Bears/""tb_"+str(i)+".png",resized_tb);
			os.remove("Teddy Bears/"+filename);
		except Exception:
			print "Whoops, failed on #" + str(i)
			continue
		continue
	cv2.imwrite("Teddy Bears/"+filename,resized_tb);
print "Done!";
