import Algorithm as al
import cv2
import database as db
#图片匹配
def match(img):
    data = db.getData()
    img1 = al.preImgge(img)
    for i in data:
        print(i[4])

        cv2.imread(i[4])
        dis = al.MHD(img1, i[4])
        print("MHD")
        print(dis)
#拍照用于注册存储
def ShotForStorage(num):
    cap = cv2.VideoCapture(1)  # 打开摄像头
    while (1):
        # get a frame
        ret, frame = cap.read()
        # show a frame
        cv2.imshow("capture", frame)  # 生成摄像头窗口
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 如果按下q 就截图保存并退出
            route = 'E:/dataimg/' + str(num) + '.jpg'
            cv2.imwrite('E:/dataimg/' + str(num) + '.jpg', frame)
            break
    cap.release()
    cv2.destroyAllWindows()
    return route

    # cap = cv2.VideoCapture(0)
    # cap.set(3, 513)  # width=1920
    # cap.set(4, 256)  # height=1080
    #
    # while (1):
    #     ret, frame = cap.read()
    #     k = cv2.waitKey(1)
    #     if k == ord('q'):
    #         break
    #     elif k == ord('s'):
    #         route='E:/dataimg/' + str(num) + '.jpg'
    #         cv2.imwrite('E:/dataimg/' + str(num) + '.jpg', frame)
    #         cv2.destroyAllWindows()
    #         cap.release()
    #     cv2.imshow("capture", frame)
    #
    #
    # return route;
def Storage(id,name,sex,age,pics):
    route='E:/dataimged/' + str(id) + '.jpg'
    picss=al.preImgge(pics)
    # #很可能存在问题

    cv2.imwrite(route, picss)
    db.inputData(id, name, sex, age, route)
#拍照进行识别
def ShotForRec():
    cap = cv2.VideoCapture(1)  # 打开摄像头
    while (1):
        # get a frame
        ret, frame = cap.read()
        # show a frame
        cv2.imshow("capture", frame)  # 生成摄像头窗口
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 如果按下q 就截图保存并退出
            route = 'E:/ToBeUIdentified/' + 1 + '.jpg'
            cv2.imwrite('E:/ToBeUIdentified/' + 1 + '.jpg', frame)
            break
    cap.release()
    cv2.destroyAllWindows()
    match(route)

