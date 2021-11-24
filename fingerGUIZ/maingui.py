# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'maingui.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(748, 503)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("master.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setStyleSheet("\n"
"QWidget\n"
"{\n"
"    background-color:rgb(54,54,54);\n"
"}\n"
" \n"
"QLabel,QGroupBox\n"
"{\n"
"    color:white;\n"
"    font:12px;\n"
"}\n"
" \n"
"QLineEdit,QPlainTextEdit\n"
"{\n"
"    color:white;\n"
"    font:13px;\n"
" \n"
"    /*边界线 border:none 没有边界*/\n"
"    border:1px solid rgb(128, 138, 135);\n"
" \n"
"    /*背景的颜色*/\n"
"    background: rgb(54, 54, 54);\n"
" \n"
"    /*边角4像素圆滑*/\n"
"    border-radius: 4px;\n"
"}\n"
" \n"
"/*鼠标滑动到LineEditor时*/\n"
"QLineEdit::hover\n"
"{\n"
"      color:rgb(250,250,250);  /*字体的颜色*/\n"
"      border-color:rgb(50,480,40);\n"
"      background-color:rgb(47,79,79);\n"
"}\n"
" \n"
"QPushButton\n"
"{\n"
"    background-color:rgb(128, 138, 135);\n"
"    color:white;\n"
"    font:16px;\n"
"    border-radius:6px; \n"
"}\n"
" \n"
"QPushButton:hover\n"
"{\n"
"    color:#0000ff;\n"
"    background-color:rgb(210, 205, 205); /*改变背景色*/\n"
"    border-style:inset;/*改变边框风格*/\n"
"    padding-left:1px;\n"
"    padding-top:1px;\n"
"}")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.btn1 = QtWidgets.QPushButton(self.centralwidget)
        self.btn1.setGeometry(QtCore.QRect(90, 130, 200, 200))
        self.btn1.setMinimumSize(QtCore.QSize(200, 200))
        self.btn1.setMaximumSize(QtCore.QSize(200, 200))
        self.btn1.setObjectName("btn1")
        self.btn2 = QtWidgets.QPushButton(self.centralwidget)
        self.btn2.setGeometry(QtCore.QRect(430, 130, 200, 200))
        self.btn2.setMinimumSize(QtCore.QSize(200, 200))
        self.btn2.setMaximumSize(QtCore.QSize(200, 200))
        self.btn2.setObjectName("btn2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 748, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.btn1.clicked.connect(MainWindow.slot1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "欢迎使用指静脉识别系统"))
        self.btn1.setText(_translate("MainWindow", "注册"))
        self.btn2.setText(_translate("MainWindow", "登录"))
import image_rc
