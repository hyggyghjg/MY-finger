import cv2

import Algorithm as al
def compare(img1,img2):
    img = cv2.imread(img1, 0)

    cv2.imshow("原图片",img)
    enhance_img = al.clahe_gabor(img)


    cv2.imshow("均质化后图片", enhance_img)
    img5=al.preImgge(img1)

    img6=al.preImgge(img2)
    cv2.imshow("处理后图片",img5)
    cv2.waitKey()
    dis=al.MHD(img5,img6)
    print("MHD")
    print(dis)
    if dis<10:
        print("成功匹配")
    else:
        print("匹配失败")
compare("E:/imgggg/1-1.png","E:/imgggg/1-2.png")
compare("E:/fff/F0101.bmp","E:/fff/F0502.bmp")