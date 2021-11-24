import sys
import Things
from Register import Ui_MainWindow as zhuceGUI
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox
from maingui import Ui_MainWindow as mainGUI
from Login import Ui_MainWindow as dengluGUI


class Ui_main(QMainWindow, mainGUI):  # 主界面
    def __init__(self):
        super(Ui_main, self).__init__()
        self.setupUi(self)

    def slot1(self):
        print("已连接")


class Ui_zhuce(QMainWindow, zhuceGUI):  # 注册界面

    def __init__(self):
        super(Ui_zhuce, self).__init__()
        self.setupUi(self)

    def slot1(self):
        self.le4.setText(Things.ShotForStorage(self.l5.text()))
        print("已连接拍摄图片")
#问题：每次进来l5不能都是一样的（1）
    def slot2(self):
        if self.le1.text()=='' or self.le2.text()=='' or self.le3.text()=='':
            QMessageBox.information(self, "标题", "信息输入不全，请重新输入!",QMessageBox.Yes)
        else:
            Things.Storage(self.l5.text(), self.le1.text(), self.le3.text(), self.le2.text(), self.le4.text())
            le1file=self.le1.text()

            t=int(self.l5.text())   #编号加一
            t=t+1
            self.l5.setText(str(t))


class Ui_denglu(QMainWindow, dengluGUI):  # 登录界面
    def __init__(self):
        super(Ui_denglu, self).__init__()
        self.setupUi(self)

    def slot1(self):
        print("已连接拍摄按钮")
        Things.ShotForRec()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    a = Ui_main()
    a.show()
    b = Ui_zhuce()
    c = Ui_denglu()
    a.btn1.clicked.connect(
        lambda: { b.show()}
    )
    a.btn2.clicked.connect(
        lambda: { c.show()}
    )
    sys.exit(app.exec_())
