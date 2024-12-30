import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
df=pd.read_csv('Hot show date(1).csv')#引入数据
plt.rcParams["font.sans-serif"]=["Simhei"]#字体黑体
print(df)
#
# #1
# df=pd.DataFrame({"Name":["XinHanCanLan","Canglan Jue","ChengXiangRuXie","FaZui","MeiGuiZhiZhan","XingFuDaoWanJia",
#                          "TianCaiJiBenFa","HuanLeSong","MengHuaLu","BingYuHuo","ErShiBuHuo","QingJun",
#                          "GuanYuTangYiShengDeYiQie","LiangGeRenDeXiaoSenLing"],
#                    "Effective playback volume":[4340,5383,3064,2065,2805,2965,2554,2542,4269,2838,2107,2567,2005,2588],
# })
#
# df.plot(x="Name",y="Effective playback volume",color="pink",kind="bar")#种类
# plt.title("Effective playback volume")
# plt.savefig("1.jpg")
# plt.show()
#
# #重新读文件
# df=pd.read_csv('D:\桌面\python\Hot show date.csv')#引入数据
# #2
# x=df["Name of episode"]
# y1=df["Market share"]
# y2=df["Number of fans"]
# y3=df["Microblog comment area"]
# #设置画布大小
# plt.figure(figsize=(10,8))
# #第一条折线图
# plt.plot(x,y1,color="pink",label="Market share",linewidth=2.0,linestyle="-")#宽度、样式
# #第二条折线图
# plt.plot(x,y2,color="cyan",label="Number of fans",linewidth=2.0,linestyle="-")
# #第三条折线图
# plt.plot(x,y3,color="gold",label="Microblog comment area",linewidth=2.0,linestyle="-")
# #添加x轴y轴标签
# plt.xlabel(u'x_Name of episode',fontsize=20)
# plt.xlabel(u'y_Number',fontsize=20)
# #标题
# plt.title(u"Hot show date",fontsize=20)
# plt.savefig("2.jpg")
# plt.show()
#
# #3
# x=df["Microblog comment area"]
# y=df["Name of episode"]
# plt.scatter(x,y,s=10,color="deepskyblue",alpha=1,linewidths=2)#s散点图点的大小，alpha散点透明度，linewidths散点边界线宽度
# #添加轴标签和标题
# plt.xlabel("Name of episode")
# plt.ylabel("Microblog comment area")
# plt.title("comment area")
# plt.savefig("3.jpg")
# plt.show()
#
# #4
# plt.figure(figsize=(6,6),dpi=90)
# plt.figure(1)
# ax1=plt.subplot(2,2,1)#行，列，象限
# plt.xlabel("Number of fans")
# plt.ylabel("Market share")
# x=df["Number of fans"]
# y=df["Market share"]
plt.plot([10,30,50,60,70,80],[10,60,30,40,110,20],color="yellow",linestyle="--")
# ax2=plt.subplot(223)
# plt.xlabel("Microblog comment area")
# plt.ylabel("Microblog likes")
# x=df["Microblog comment area"]
# y=df["Microblog likes"]
# plt.plot([5,10,15,40],[5,10,15,200],color="r",linestyle="-")
# ax3=plt.subplot(122)
# plt.xlabel("Microblog likes")
# plt.ylabel("Number of fans")
# x=df["Microblog likes"]
# y=df["Number of fans"]
# plt.plot([5,10,15,200],[10,30,50,60],color="yellowgreen",linestyle="-.")
# plt.title("Audience likes TV plays")
# plt.savefig("4.jpg")
# plt.show()


df=pd.read_csv('Hot show date(1).csv')#引入数据
#3
date=df["Total number of sets"]
label=df["Name of episode"]
#设置每块区域离圆心的距离
explode=[0.05,0.03,0.02,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.02]
colors=["pink","aqua","teal","blue","greenyellow","magenta","gold","lightcoral","salmon","cyan",
        "deeppink","orchid","m","plum"]
plt.pie(date,labels=label,explode=explode,autopct="%.2f%%",colors=colors)#加上百分数，.2f保留两位小数
plt.title("Total number of sets")
plt.savefig("3.jpg")
plt.show()










