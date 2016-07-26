import cv2
import os
i = 0;
while(i < 3001):
	filename = "Towels/towel_"+str(i);
	try:
		os.rename(filename,filename +".jpg")
	except Exception:
		print "Error on number: " + str(i)
	finally:
		i = i+1
print "Done!";
