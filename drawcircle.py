import cv2
 
image = cv2.imread('brainTumor.png')
x=100
y=100
r=50
cv2.circle(image,(x, y), r, (0,255,0))
cv2.imshow('brainTumor image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()