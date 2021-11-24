import cv2
import Algorithm as al

img = cv2.imread("111.jpg",-1)

img = al.get_ROImm(img);
cv2.namedWindow("Image",0);
cv2.resizeWindow("Image", 480, 680);
cv2.imshow("Image", img)
cv2.waitKey (0)
cv2.destroyAllWindows()
