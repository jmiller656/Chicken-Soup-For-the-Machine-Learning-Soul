import flickr_api
import urllib, urlparse
import os
import sys
from flickr_api import *
flickr_api.set_keys(api_key='59ac5a1d9e86df4ecc6d96e49f94e27c', api_secret='5a4bad181d5904f4')
i = 8
w = Walker(Photo.search, tags="bagel",text="bagel",sort="relevance")
for photo in w:
    try:
    	url = photo.getPhotoFile()
    	f = open("bagel_" + str(i),'wb')
    	f.write(urllib.urlopen(url).read())
    	f.close()
    	i = i+1
    	if i > 3000:
        	break
    except IOError:
	print("Whoops, failed on number: "+str(i))
