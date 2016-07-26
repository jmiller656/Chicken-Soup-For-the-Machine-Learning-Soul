import cv2
import os
i = 65
while(i < 3000):
	filename = "Dogs/dog_"+str(i);
	os.rename(filename,filename +".jpg")
	i = i+1
print "Done!";
