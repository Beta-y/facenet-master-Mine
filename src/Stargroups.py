# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 11:01:41 2019

@author: Beta
"""

import cv2
import math 
import numpy as np

#规定x为横向，y为纵向
#Beginpointx = 15 #第一个迭代的点的列号
#Beginpointy = 10 #第一个迭代的点的行号

def Findcenter(L_eyeimg,Beginpointx,Beginpointy,Tres):
    #L_eyeimg = cv2.imread('E:\\TensorCode\\facenet-master-Mine\\src\\eye.jpg')
    #归一化为20*30
    L_eyesize = cv2.resize(L_eyeimg, (30,20), interpolation=cv2.INTER_CUBIC ) #缩放为30*180
    #灰度化
    #L_eyesize = L_eyeimg
    L_eyegray = cv2.cvtColor(L_eyesize,cv2.COLOR_BGR2GRAY)
    #高斯滤波
    Blackimg = cv2.GaussianBlur(L_eyegray,(3,3),0)
    
    trs,_ = cv2.threshold(Blackimg, 20, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)  # Otsu 滤波
    _,img_Guassian = cv2.threshold(Blackimg,5*trs/7,255,cv2.THRESH_BINARY)
                    
    #画出第一个圆
    #cv2.circle(L_eyesize,(Beginpointx,Beginpointy), 0, (0,255,255), 0)
    #cv2.circle(L_eyesize,(10,10), 1, (0,0,255), 0)
    
    #以中心点为起始点，发出18条射线，找出梯度改变的点
    #Tres = 30
    countsum = 0 #点数
    PreparePoint = np.zeros((18,4))  #第一轮备选点
    for line in range(0,18):
        theta = line * math.pi / 9
        x_pre = Beginpointx
        y_pre = Beginpointy
        while True:
            x_now = x_pre + math.cos(theta)
            y_now = y_pre - math.sin(theta)
            if x_now < 1 or x_now > 29 or y_now < 1 or y_now > 19:
                break
            Delta = int(img_Guassian[int(y_now),int(x_now)]) - int(img_Guassian[int(y_pre),int(x_pre)])
            if Delta > Tres :
                if countsum == 0 or (PreparePoint[countsum - 1,0] != int(x_now) and PreparePoint[countsum - 1 ,1] != int(y_now) ):
                   #L_eyesize[int(y_now),int(x_now)] = (0,255,0) 
                   PreparePoint[countsum,0] = int(x_now)
                   PreparePoint[countsum,1] = int(y_now)
                   PreparePoint[countsum,2] = Delta
                   PreparePoint[countsum,3] = theta
                   countsum += 1
                break
            x_pre = x_now
            y_pre = y_now
    #cv2.imshow("First",L_eyesize)
   # print(PreparePoint) 
   # print(countsum)            
    
    #以每一个梯度改变的点为中心点，发出左右各50度的射线
    #n = 5*Delta/Tres n为射线条数，最少为5条
    #delta = 100 / (n-1) 射线间隔角度
    countsum2 = 0
    PreparePoint2 = np.zeros((10*countsum,2))  #第一轮备选点
    for count in range(0,countsum): #共countsum个点
        n = int(5*PreparePoint[count,2]/Tres)
        delta = int(100 / (n-1))
        for i in range(0,n): #共n条射线
            theta =  -math.pi  + PreparePoint[count,3] + 50/180 * math.pi - i*delta
            
            x_pre = PreparePoint[count,0]
            y_pre = PreparePoint[count,1]
            while True:
                x_now = x_pre + math.cos(theta)
                y_now = y_pre - math.sin(theta)
                if x_now < 1 or x_now > 29 or y_now < 1 or y_now > 19:
                    break
                Delta = int(img_Guassian[int(y_now),int(x_now)]) - int(img_Guassian[int(y_pre),int(x_pre)])
                if Delta > Tres :
                    if countsum2 == 0 or (PreparePoint2[countsum2 - 1,0] != int(x_now) and PreparePoint2[countsum2 - 1 ,1] != int(y_now)):
                       L_eyesize[int(y_now),int(x_now)] = (0,0,255) 
                       PreparePoint2[countsum2,0] = int(x_now)
                       PreparePoint2[countsum2,1] = int(y_now)
                       countsum2 += 1
                    break
                x_pre = x_now
                y_pre = y_now
    cv2.imshow("Second",L_eyesize)
   # print(PreparePoint2)     
    #print(countsum2) 
    
    sumx = 0
    sumy = 0
    for i in range(0,countsum2):
        sumx += PreparePoint2[i,0]
        sumy += PreparePoint2[i,1]
    avex = int(sumx/(countsum2+0.01))
    avey = int(sumy/(countsum2+0.01))
    #L_eyesize[int(avey),int(avex)] = (0,255,0) 
    #CenterX = 0
    #CenterY = 0
    #Radius = 65535
    #
    #
    #RoundPoint = np.zeros((3,2))
    #RoundPoint[:,0] = PreparePoint2[0:3,0]
    #RoundPoint[:,1] = PreparePoint2[0:3,1]
    #
    #RoundPoint_temp = np.zeros((3,2))
    #RoundPoint_temp[:,:] = RoundPoint[:,:]
    #print(RoundPoint_temp[0,0])
    #x0,y0,R0 = getCircle(RoundPoint[0,0],RoundPoint[0,1],RoundPoint[1,0],RoundPoint[1,1],RoundPoint[2,0],RoundPoint[2,1])
    #
    #
    #for i in range(3,countsum2):   
    #    for j in range(0,3):
    #        RoundPoint_temp[j,:] = PreparePoint2[i,:]
    #        x1,y1,R1 = getCircle(RoundPoint_temp[0,0],RoundPoint_temp[0,1],RoundPoint_temp[1,0],RoundPoint_temp[1,1],RoundPoint_temp[2,0],RoundPoint_temp[2,1])
    #        if Inornot(RoundPoint[j,0],RoundPoint[j,1],x1,y1,R1) == 1 and R1 < Radius:
    #            Radius = R1
    #            CenterX = x1
    #            CenterY = y1
    #            RoundPoint[j,:] = RoundPoint_temp[j,0]
    #            print(CenterY,CenterX,Radius)
    #        else:
    #            RoundPoint_temp[j,:] = RoundPoint[j,:] #还原
    #
    ##L_eyesize[int(CenterY),int(CenterX)] = (0,255,0) 
    #
    #print(CenterY,CenterX,Radius)
    #cv2.circle(L_eyesize,(int(CenterX),int(CenterY)), int(Radius), (0,255,255), 1)
    
    
   # L_eyesize = cv2.resize(L_eyesize, (300,200), interpolation=cv2.INTER_CUBIC ) #缩放为30*180
    
    #cv2.imshow("原图",L_eyesize)
    #cv2.imshow("灰度图",L_eyegray)
   # cv2.imshow("高斯滤波",img_Guassian)
    if avex >= 30 or avex == 0:
        avex = Beginpointx
    if avey >= 20 or avey == 0:
        avey = Beginpointy
    return avex,avey,PreparePoint2,countsum2
    #if cv2.waitKey(0) & 0xFF == ord('q'):
     #   cv2.destroyAllWindows()
