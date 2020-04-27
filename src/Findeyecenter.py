# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 19:14:27 2019

@author: Beta
"""
import cv2
import random
from Stargroups import Findcenter
from Round_radius import getCircle
from Round_radius import Distance
from Filter import Filtercol

xpre = 10
ypre = 10
L_eyeimg = cv2.imread('E:\\TensorCode\\facenet-master-Mine\\src\\eye.jpg')
L_eyesize = cv2.resize(L_eyeimg, (30,20), interpolation=cv2.INTER_CUBIC ) #缩放为30*20

#L_eyegray = cv2.cvtColor(L_eyesize,cv2.COLOR_BGR2GRAY)
#img_Guassian = cv2.GaussianBlur(L_eyegray,(3,3),0)
#trs,_ = cv2.threshold(img_Guassian, 20, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)  # Otsu 滤波
#_,Blackimg = cv2.threshold(img_Guassian,5*trs/7,255,cv2.THRESH_BINARY)


for i in range(0,10):
    x,y,Preparepoint,countsum = Findcenter(L_eyeimg,int(xpre),int(ypre),255) 
   # print(x,y,i)
    if (xpre - x)*(xpre - x) + (ypre - x)*(ypre - y) <= 9 and i > 2:
        break   
    if x != 0 and y != 0:
        xpre = x
        ypre = y

maxcount = 0
for circletime in range(0,2*countsum):
    #随机获取三个点
    P0 = random.randint(0,countsum - 1)
    P1 = random.randint(0,countsum - 1)
    P2 = random.randint(0,countsum - 1)
    #求外接圆
    x0,y0,r = getCircle(Preparepoint[P0,0],Preparepoint[P0,1],Preparepoint[P1,0],Preparepoint[P1,1],Preparepoint[P2,0],Preparepoint[P2,1])
    
    if r > 4 and r < 6.5 and x0 > 0 and y0 > 0 and Distance(xpre,ypre,x0,y0) < r:
        #print(x0,y0,r)
        #L_eyesize[y0,x0] = (0,0,255)
        count = 0
        for lefttimes in range(0,countsum):
            if lefttimes != P0 and  lefttimes != P1 and lefttimes != P0:  #剩余的点
                dis = Distance(Preparepoint[lefttimes,0],Preparepoint[lefttimes,1],x0,y0) #求剩余点到圆心的距离
                if dis - r < 2: #如果距离接近则计数
                    count += 1
        if count > maxcount:
            Finalx = x0
            Finaly = y0
            Finalr = r
            maxcount = count 
cv2.circle(L_eyesize,(Finalx,Finaly),Finalr,(255,0,0),1,0,0)
print(maxcount)
L_eyesize[Finaly,Finalx]  = (0,255,0)                
    
#for count in range(0,countsum):
#   L_eyesize[int(Preparepoint[count,1]),int(Preparepoint[count,0])] = (0,0,255)
##print(Preparepoint)
img = cv2.resize(L_eyesize, (300,200), interpolation=cv2.INTER_CUBIC ) #缩放为30*180

cv2.imshow("a",img)
#L_eyesize[ypre,xpre] = (0,255,0)
#L_eyesize = cv2.resize(L_eyesize, (300,200), interpolation=cv2.INTER_CUBIC ) #缩放为30*180
#cv2.imshow("aa",L_eyesize)
cv2.waitKey(0)
cv2.destroyAllWindows()