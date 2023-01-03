import cv2
import numpy as np
image=cv2.imread('3.jpg')

"""detector=cv2.SimpleBlobDetector_create()
keypoints=detector.detect(image)
blank=np.zeros((1,1))
blobs=cv2.drawKeypoints(image,keypoints,blank,(0,255,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
number_of_blobs=len(keypoints)
text="total no of blobs"+str(len(keypoints))
cv2.putText(blobs,text,(50,650),cv2.FONT_HERSHEY_SIMPLEX,1,(100,0,255),2)"""


#initialize parameter setting using cv2.SimpleBlobDetector
params=cv2.SimpleBlobDetector_Params()

params.filterByArea=True
params.minArea=0.01

params.filterByCircularity=True
params.minCircularity=0.01


params.filterByConvexity=False
params.minConvexity=2

detector=cv2.SimpleBlobDetector_create(params)

keypoints=detector.detect(image)



blank=np.zeros((1,1))
blobs=cv2.drawKeypoints(image,keypoints,blank,(0,255,0),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
number_of_blobs=len(keypoints)
text="total no of circular blobs"+str(len(keypoints))
cv2.putText(blobs,text,(20,550),cv2.FONT_HERSHEY_SIMPLEX,1,(0,100,255),2)



cv2.namedWindow('original image',cv2.WINDOW_NORMAL)
cv2.imshow('original image', image)

cv2.namedWindow('blob using default parameters',cv2.WINDOW_NORMAL)
cv2.imshow('blob using default parameters',blobs)

cv2.waitKey(0)
cv2.destroyAllWindows()