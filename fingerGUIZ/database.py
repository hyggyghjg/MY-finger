import pymysql
dbhost='localhost'
dbuser='root'
dbpass='123456'
dbname='finger'
def getData():
	try:
		db=pymysql.connect(host=dbhost,user=dbuser,password=dbpass,database=dbname)
		cursor = db.cursor()
		cursor.execute("SELECT * from finger")
	# 使用 fetchone() 方法获取一条数据
		#data = cursor.fetchone()     一条数据
		data = cursor.fetchall()    #多条数据

		print(data)
		return data
	except pymysql.Error as e:
		print("数据库连接失败："+str(e))

	finally:
		if db:
			db.close()
			print('关闭数据库连接....')

def inputData(id,name,sex,age,pics):#pics:存储图片的文件夹
	db=pymysql.connect(host=dbhost,user=dbuser,password=dbpass,database=dbname)
	cursor = db.cursor()
	sql = "insert into finger (IDCARD,NAME,SEX, AGE,PICS) values ('%s','%s','%s','%s','%s')" % \
		  (id,name,sex,age,pics)
	cursor.execute(sql)
	db.commit()
	db.close()
