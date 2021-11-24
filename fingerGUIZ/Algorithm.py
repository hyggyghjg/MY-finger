import cv2
import numpy as np
from queue import Queue
#脊波变换-->用于图像增强
#def ridgelet_transform(img):
#截取矩形兴趣域
def get_ROImm(img):
    bimg1 = cv2.Canny(img, 20, 60)
    kernel = np.ones((50, 50), np.uint8)
    # 图像膨胀
    dilate_result = cv2.dilate(bimg1, kernel)
    # 图像腐蚀
    bimg1 = cv2.erode(dilate_result, kernel)
    bimg2 = bimg1
    h, w = bimg1.shape
    y1 = 0
    y2 = 10000
    # 感兴趣提取（矩形区域）
    for k in range(h):

        for i in range(w // 2, 0, -1):
            if (bimg2[k][i] == 255 and i > y1):
                y1 = i
                break

        for j in range(w // 2, w):
            if (bimg2[k][j] == 255 and j < y2):

                y2 = j

                break


    img = img[:, y1:y2]
    return img
def get_ROI(img):
    bimg1 = cv2.Canny(img, 20, 60)
    bimg2 = bimg1
    h, w = bimg1.shape
    y1 = 0
    y2 = 0
    # 感兴趣提取（矩形区域）
    for k in range(w // 2, 0, -1):
        for i in range(10):
            if (bimg2[i][k] == 255):
                y1 = k
                break
        for i in range(h // 2, 0, -1):
            if (bimg2[i][k] == 255):
                for a in range(i - 1, 0, -1):
                    bimg2[a][k] = 0;
                break
        for j in range(h // 2, h):
            if (bimg2[j][k] == 255):
                for a in range(j + 1, h):
                    bimg2[a][k] = 0;
                break
    for k in range(w // 2, w):
        for i in range(10):
            if (bimg2[i][k] == 255):
                y2 = k
                break
        for i in range(h // 2, 0, -1):
            if (bimg2[i][k] == 255):
                for a in range(i - 1, 0, -1):
                    bimg2[a][k] = 0;
                break
        for j in range(h // 2, h):
            if (bimg2[j][k] == 255):
                for a in range(j + 1, h):
                    bimg2[a][k] = 0;
                break
    maxup = 0
    mindown = 10000
    for k in range(w):
        for i in range(h // 2, 0, -1):
            if (bimg2[i][k] == 255):
                if (i > maxup):
                    maxup = i
                    break
        for j in range(h // 2, h):
            if (bimg2[j][k] == 255):
                if (j < mindown):
                    mindown = j
                    break
    kernel = np.ones((50, 50), np.uint8)
    # 图像膨胀
    dilate_result = cv2.dilate(bimg2, kernel)
    # 图像腐蚀
    bimg2 = cv2.erode(dilate_result, kernel)
    bimg2 = bimg2[:, y1:y2]
    img=img[:, y1:y2]
    return img
# def get_ROIii(img):
#     #去除部分无关区域
#     img=img[30:226,50:380]
#     # cv2.imshow('quchu',img)
#     #提取边缘
#     bimg1=cv2.Canny(img,20,240)
#     # cv2.imshow('canny提取边缘', bimg1)
#     h,w=bimg1.shape
#     y1=0
#     y2=0
#     for k in range(w):
#         for i in range(h//2,0,-1):
#             if(bimg1[i][k]==255):
#                 y1+=i
#         for j in range(h//2,h):
#             if(bimg1[j][k]==255):
#                 y2+=j
#     y1=y1//(w)
#     y2=y2//(w)
#     roi_img=img[y1:y2,:]
#     #尺度归一化
#     # print(roi_img.shape)
#     print(roi_img.shape)
#     if roi_img.shape[0]==0 or roi_img.shape[1]==0:
#         roi_img=img[:,50:202]
#     roi_unimg=cv2.resize(roi_img,(330,144),cv2.INTER_LINEAR)
#     # cv2.imshow('uni', roi_unimg)
#     return roi_unimg
#
#
# #截取矩形兴趣域
# def get_ROI0(img):
#     #去除部分无关区域
#     img=img[0:320,200:460]
#     # clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
#     # img = clahe.apply(img)
#     img=cv2.medianBlur(img,3)
#     cv2.imshow('caij',img)
#     #提取边缘
#     bimg=cv2.Canny(img,20,120)
#     cv2.imshow('tst', bimg)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     #裁剪：分为左右两部分，分别计算平均边缘坐标
#     h,w=bimg.shape
#     x1=0
#     x2=0
#     for k in range(h):
#         for i in range(w//2,0,-1):
#             if(bimg[k][i]==255):
#                 x1+=i
#         for j in range(w//2,w):
#             if(bimg[k][j]==255):
#                 x2+=j
#     x1=x1//h
#     x2=x2//h
#     # print(x1)
#     # print(x2)
#     roi_img=img[:,x1:x2]
#     if roi_img[1]==0:
#         roi_img=img[:,50:202]
#     #尺度归一化
#     roi_unimg=cv2.resize(roi_img,(150,320),cv2.INTER_LINEAR)
#     # print("???")
#     # print(roi_img.shape)
#     return roi_unimg
#
#图像增强
def clahe_gabor(roi_img):
    #CLAHE:限制对比度的自适应直方图均衡
   # clahe_img = cv2.equalizeHist(roi_img)

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(5,5))
    clahe_img=clahe.apply(roi_img)
    # cv2.imwrite('D:/finger_vein_recognition/clahe_roi_img.jpg',clahe_img)
    #Gabor滤波和融合都没有调试好
    # res=np.zeros(roi_img.shape,np.uint8)
    # for i in range(4):
    #     gabor = cv2.getGaborKernel(ksize=(5, 5), sigma=20, theta=i*45, lambd=30, gamma=0.375)
    #     gabor_img = cv2.filter2D(src=clahe_img, ddepth=cv2.CV_8UC3, kernel=gabor)
    #     cv2.imwrite('D:/finger_vein_recognition/'+str(i)+'.jpg', gabor_img)
    #     # gabor_img=cv2.GaussianBlur(gabor_img,ksize=(3,3),sigmaX=0.8)
    #     # cv2.imshow('1', gabor_img)
    #     # ret,bimg=cv2.threshold(gabor_img,30,255,cv2.THRESH_BINARY)
    #     # cv2.imshow('2',bimg)
    #     # res=cv2.add(res,bimg)

    return clahe_img
'''
sigma_x:标准差
dnum:方向的数目
s:尺度
L:tje length od y-direction
'''

#得到多尺度匹配滤波核
def getMultiMatchFilterKernel(sigma,L,theta,s):
    width=int(np.sqrt((6*sigma+1)**2+L**2))
    mutilMatchFilter=np.zeros((width,width))
    if np.mod(width,2)==0:
        width=width+1
    halfL=int((width-1)/2)
    row=1
    for y in range(halfL,-halfL,-1):
        col=1
        for x in range(-halfL,halfL):
            p=x*np.cos(theta)+y*np.sin(theta)
            q=x*np.cos(theta)-y*np.sin(theta)
            if np.abs(p)>3*sigma or np.abs(q)>s*L/2:
                mutilMatchFilter[row][col]=0
            else:
                # mutilMatchFilter[row][col]=-np.exp(-(p**2)/(s*sigma**2))
                mutilMatchFilter[row][col] = -np.exp(-5*(p/sigma)**2/(np.sqrt(2*np.pi)*sigma*s))
            col=col+1
        row=row+1
    mean=np.sum(mutilMatchFilter)/np.count_nonzero(mutilMatchFilter)
    mutilMatchFilter[mutilMatchFilter!=0]=mutilMatchFilter[mutilMatchFilter!=0]-mean
    return mutilMatchFilter
def applyMultiMatchFilter(img,sigma_x,L,dnum,s):
    h,w=img.shape
    mf_img=np.zeros((h,w,dnum),dtype=np.uint8)
    # res = np.zeros((h, w, dnum), dtype=np.uint8)
    for i in range(dnum):
        multiMatchFilter=getMultiMatchFilterKernel(sigma_x,L,(np.pi/dnum)*i,s)
        # print(multiMatchFilter)
        mf_img[:,:,i]=cv2.filter2D(img,ddepth=cv2.CV_8UC3,kernel=multiMatchFilter)
    # print(mf_img.shape)
    res=np.max(mf_img,axis=2)
    return res

def MMF(enhance_img):
    # 应用多尺度匹配滤波提取静脉纹路
    mutil_img1 = applyMultiMatchFilter(enhance_img, 5, 5, 12, 0.03)
    # cv2.imshow('s=0.03 filter response', mutil_img1)
    # cv2.imwrite('D:/finger_vein_recognition/003.jpg',mutil_img1)
    mutil_img2 = applyMultiMatchFilter(enhance_img, 5, 5, 12, 0.06)
    # cv2.imshow('0.06 filter response', mutil_img2)
    # cv2.imwrite('D:/finger_vein_recognition/006.jpg',mutil_img2)
    mutil_img3 = applyMultiMatchFilter(enhance_img, 5, 5, 12, 0.09)
    # cv2.imshow('0.09 filter response', mutil_img3)
    # 三个尺度加权乘积
    res = cv2.multiply(mutil_img1, mutil_img2, scale=0.1)
    res = cv2.multiply(res, mutil_img3, scale=0.1)
    #进行形态学处理
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel, iterations=1)
    res = cv2.medianBlur(res, 5)
    # cv2.imshow('multi-scale matched filter response', res)
    #二值化
    ret, res = cv2.threshold(res, 40, 255, cv2.THRESH_BINARY)
    res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel, iterations=1)
    # cv2.imshow('bi', res)
    return res

#细化 zhng-suen细化算法
# 定义像素点周围的8邻域
#                P9 P2 P3
#                P8 P1 P4
#                P7 P6 P5
def neighbours(x, y, image):
    img = image
    x_1, y_1, x1, y1 = x - 1, y - 1, x + 1, y + 1
    return [img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],  # P2,P3,P4,P5
            img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1]]  # P6,P7,P8,P9

# 计算邻域像素从0变化到1的次数
def transitions(neighbours):
    n = neighbours + neighbours[0:1]  # P2,P3,...,P8,P9,P2
    return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))  # (P2,P3),(P3,P4),...,(P8,P9),(P9,P2)

def delete(img,flag):
    rows, cols = img.shape
    for y in range(cols):
        for x in range(rows):
            if flag[x][y]==1:
                img[x,y]=0
    return img

def ZhangSuen(img):
    flag = np.zeros(img.shape)
    rows,cols=img.shape
    #step one
    for y in range(1,cols-1):
        for x in range(1,rows-1):
            if img[x][y]==1:#前景点
                P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, img)
                if(2<=sum(n)<=6 and transitions(n)==1 and
                        P2*P4*P6==0 and  P4*P6*P8==0):
                    flag[x,y]=1
    if np.sum(flag)>0:
        img=delete(img,flag)
        #flag清零
        flag = np.zeros(img.shape)
        #step two
        for y in range(1,cols -1) :
            for x in range(1,rows -1):
                if img[x][y] == 1:  # 前景点
                    P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, img)
                    if (2 <= sum(n) <= 6 and transitions(n) == 1 and
                            P2 * P4 * P8 == 0 and P2 * P6 * P8 == 0):
                        flag[x, y] = 1
        if np.sum(flag)>0:
            img=delete(img,flag)
            img0=ZhangSuen(img)
            return img
        else:
            return img
    else:
        return img


def findBurr(img,i,j,q):
    if q.full():
        while(not q.empty()):
            index=q.get()
            img[index[0]][index[1]]=1
        # print(q.empty())
        return img
    else:
        n = neighbours(i, j, img)
        if sum(n) > 1:
            return img
        else:
            if sum(n) == 0:
                img[i][j] = 0
                return img
            else:
                q.put([i, j])
                img[i][j] = 0
                for x in range(i - 1, i + 2):
                    for y in range(j - 1, j + 2):
                        if img[x][y] == 1:
                            i = x
                            j = y
                            img=findBurr(img,i,j,q)
                            return img
#细化的毛刺去除
def removeBurr(img):
    rows,cols=img.shape
    # print(rows,cols)
    thresh=25
    q = Queue(thresh)
    for i in range(1,rows-1):
        for j in range(1,cols-1):
            # print(img[i][j])
            # print(i,j)
            if img[i][j]==1:
                img=findBurr(img,i,j,q)
                while (not (q.empty())):
                    q.get()

    img = black(img)
    return img

#
# #先进先出队列
# q=Queue(maxsize=5)

#四周全黑
def black(img):
    rows,cols=img.shape
    for i in range(rows):
        img[i][0]=0
        img[i][cols-1]=0
    for j in range(cols):
        img[0][j]=0
        img[rows-1][j]=0
    return img

def enhanceImage(filename):
    img = cv2.imread(filename, 0)
    roi_img = get_ROI(img)
    # 使用clahe图像增强并中值滤波
    enhance_img = clahe_gabor(roi_img)
    enhance_img = cv2.medianBlur(enhance_img, 3)
    # cv2.imshow('enhance', enhance_img)
    return enhance_img

def preImgge(filename):
    img = cv2.imread(filename, 0)
#    roi_img = get_ROI(img)
    # 使用clahe图像增强并中值滤波
    ####################################################################测试
    enhance_img = clahe_gabor(img)
  #  enhance_img = cv2.medianBlur(enhance_img, 3)
    # cv2.imshow('enhance', enhance_img)
    #多尺度匹配滤波
    #####################################$
    res = MMF(enhance_img)
    res = res / 255
    #细化
    xihua = ZhangSuen(res)
    # cv2.imshow('Zhangsuen', xihua)
    #细化毛刺去除
    remove = removeBurr(xihua)
    return remove*255

import cv2
import numpy as np
from matplotlib import pyplot as plt
#LBP
# def LBP(img):
#     rows,cols=img.shape
#     lbp=np.zeros(img.shape,dtype=np.uint8)
#     for i in range(1,rows-1):
#         for j in range(1,cols-1):
#             for x in range(i-1,i+2):
#                 for y in range(j-1,j+2):
#                     if img[x][y]>img[i][j]:
#                         lbp[x][y]=255
#                     else:
#                         lbp[x][y]=0
#     return lbp
# def LBP_match(img,img2):
#     # img = cv2.imread(filename1, 0)
#     # img2=cv2.imread(filename2, 0)
#     lbp=LBP(img)
#     lbp2=LBP(img2)
#     score=1-(np.sum(lbp^lbp2))/(330*144*255)
#     # print(score)
#     # cv2.imshow("lbp",lbp)
#     # cv2.imshow("lbp2",lbp2)
#     return score
#
# #img1,img2为同一个体的两份指静脉图像样本
# def PBBM(img1,img2):
#     lbp1=LBP(img1)
#     lbp2=LBP(img2)
#     # lbp3=LBP(img3)
#     pbbm=np.zeros(img1.shape,dtype=np.uint8)
#     rows,cols=img1.shape
#     for i in range(rows):
#         for j in range(cols):
#             if lbp1[i][j]==lbp2[i][j]:
#                 # if lbp3[i][j]==lbp2[i][j]:
#                  pbbm[i][j]=lbp1[i][j]
#             else:
#                 pbbm[i][j]=-1
#     return pbbm
# def PBBM_match(lbp,pbbm):
#     # lbp=LBP(img1)
#     # pbbm=PBBM(img2,img3)
#     rows, cols = lbp.shape
#     score=0
#     count=0
#     for i in range(rows):
#         for j in range(cols):
#             if pbbm[i][j] == -1:
#                 continue
#             else:
#                 count=count+1
#                 if lbp[i][j]==pbbm[i][j]:
#                     score=score+1
#     print(score)
#     print(count)
#     score=score/count
#     return score
#基于细节点和MHD算法


'''
Opencv中的函数cv2.goodFeatureToTrack()用来进行Shi-Tomasi角点检测，其参数说明如下所示：
第一个参数：通常情况下，其输入的应是灰度图像；
第二个参数N：是想要输出的图像中N个最好的角点；
第三个参数：设置角点的质量水平，在0~1之间；代表了角点的最低的质量，小于这个质量的角点，则被剔除；
最后一个参数：设置两个角点间的最短欧式距离；也就是两个角点的像素差的平方和；
'''
def corner(img):
    corners = cv2.goodFeaturesToTrack(img, 30, 0.5, 10)  # 返回的结果是 [[ a., b.]] 两层括号的数组。
    # print(corners)
    corners = np.int0(corners)
    # print(corners)
    for i in corners:
        x, y = i.ravel()
        print((x,y))
        cv2.circle(img, (x, y), 4, 255, -1)  # 在角点处画圆，半径为2，红色，线宽默认，利于显示
    cv2.imshow('imgcor',img)
    return corners

def end_cross_point(img2):
    rows,cols=img2.shape
    # print(img2.shape)
    tpoint=[]
    img=img2.copy()
    img=img/255
    img=img.astype(int)
    for i in range(1,rows-1):
        for j in range(1,cols-1):
            if img[i][j]==1:
                # val=(np.abs(img[i-1][j]-img[i-1][j-1])+np.abs(img[i-1][j+1]-img[i-1][j])+np.abs(img[i][j+1]-img[i-1][j+1])+np.abs(img[i+1][j+1]-img[i][j+1])+np.abs(img[i+1][j]-img[i+1][j+1])+np.abs(img[i+1][j-1]-img[i+1][j])+np.abs(img[i][j-1]-img[i+1][j-1])+np.abs(img[i-1][j-1]-img[i][j-1]))
                p1=img[i-1][j-1]
                p2=img[i-1][j]
                p3=img[i-1][j+1]
                p4=img[i][j+1]
                p5=img[i+1][j+1]
                p6=img[i+1][j]
                p7=img[i+1][j-1]
                p8=img[i][j-1]
                val=np.abs(p2-p1)+np.abs(p3-p2)+np.abs(p4-p3)+np.abs(p5-p4)+\
                    np.abs(p6-p5)+np.abs(p7-p6)+np.abs(p8-p7)+np.abs((p1-p8))
                if val==2 or val>=6:
                    tpoint.append([i,j])
                else:
                    continue

    return tpoint

#输入细化后的图像
def MHD(img1, img2):
    score = 0
    point1=end_cross_point(img1)
    point2=end_cross_point(img2)
    N=len(point1)
    dis=0
    for i in range(N):
        d0=np.sqrt(np.power((point1[i][0]-point2[0][0]),2)+np.power((point1[i][1]-point2[0][1]),2))
        for j in range(1,len(point2)):
            d=np.sqrt(np.power((point1[i][0]-point2[j][0]),2)+np.power((point1[i][1]-point2[j][1]),2))
            if d<d0:
                d0=d
        dis=dis+d0
    dis=dis/N
    return dis
#匹配函数
# def match(num1,num2):
#     # filename1="D:/finger_vein_recognition/data/"+num1+".bmp"
#     # filename2 = "D:/finger_vein_recognition/data/" + num2 + ".bmp"
#     filename1 = "num1.png"
#     filename2 = "num2.png"
#     print(filename1)
#     num1=int(num1)
#     num2=int(num2)
#
#     ismatch = False
#     # 真实匹配情况
#     if num1 % 40 == num2 % 40:
#         ismatch = True
#     print(ismatch)
#     num3=str((num1+40)%120)
#     # num4=str((num1+80)%120)
#     filename3 = "D:/finger_vein_recognition/data/" + num3 + ".bmp"
#     # filename4 = "D:/finger_vein_recognition/data/" + num4 + ".bmp"
#
#
#     ##LBP
#     img1=enhanceImage(filename1)
#     img2=enhanceImage(filename2)
#     LBP_score=LBP_match(img1,img2)
#     print("LBP:")
#     print(LBP_score)
#
#     ##PBBM
#     img3=enhanceImage(filename3)
#     # img4=enhanceImage(filename4)
#     pbbm=PBBM(img1,img3)
#     lbp=LBP(img2)
#     PBBM_score=PBBM_match(lbp,pbbm)
#     print("PBBM")
#     print(PBBM_score)
#
#     ##细节点MHD
#     img5=preImgge(filename1)
#     img6=preImgge(filename2)
#     dis=MHD(img5,img6)
#     print("MHD")
#     print(dis)
#     return LBP_score,PBBM_score,dis,ismatch
#
#计算系统准确率
#score>70
#distance<12
#认为是匹配的
def is_match_score(score):
    if score>0.7:
        return True
    else:
        return False
def is_match_dis(dis):
    if dis<15:
        return True
    else:
        return False
def vote(lbp,pbbm,dis):
    if (is_match_score(lbp) and is_match_score(pbbm)) or (is_match_score(lbp) and is_match_dis(dis)) or (is_match_score(pbbm) and is_match_dis(dis)):
        return True
    else:
        return False

