# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 22:40:32 2019

@author: shinelon
"""

#  从视频流中采集目标对象的人脸，用于训练分类模型，这个模型用于主程序中摄像头实时视频中人脸识别出目标对象
#  每1帧采集一张人脸，一共采集100个目标对象的样本，使用mtcnn对采集帧进行人脸检测和对齐
#import math
import cv2
#import sys
#import os
import tensorflow as tf
#import numpy as np
import align.detect_face
#import facenet
#import matplotlib.pyplot as plt
from Filter import Filtercol
from Filter import Findboundcol
from Filter import Findboundrow

import socket
video_capture = cv2.VideoCapture(0)
#video_capture = cv2.VideoCapture("E:\\TensorCode\\facenet-master-Mine\\src\\Test.mp4")


capture_interval = 1
capture_num = 100
capture_count = 0
frame_count = 0
detect_multiple_faces = False #因为是训练目标对象，一次只有一张人脸
 
#这里引用facenet/src/align/align_dataset_mtcnn.py文件的代码对采集帧进行人脸检测和对齐
minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor
        
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

Eyex = 15
Eyey = 10

#连接客户端
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("localhost",8888))
server.listen(5)
connection, address = server.accept()
#print(connection, address)
msg = ""
eyeX = 0
eyeY = 0
cacutime = 0
SuccessFlag = 0
while True:
    ret, frame = video_capture.read()
    try:
        frame = cv2.flip(frame,1)#水平翻转
    #    cv2.imshow("原图", frame)
        h=frame.shape[0]
        w=frame.shape[1]
        #每1帧采集一张人脸，这里采样不进行灰度变换，直接保存彩色图

        if(capture_count%capture_interval == 0): 
            
            bounding_boxes, points = align.detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
            nrof_faces = bounding_boxes.shape[0]
    #        print(points.shape) #摄像头原图像尺寸

            
            for face_position in bounding_boxes: #因为只采集一张人脸，所以实际只遍历一次
                SuccessFlag = 1
                face_position=face_position.astype(int)
                cropped = frame[face_position[1]:face_position[3],face_position[0]:face_position[2],:]

                if(len(points) == 10 and len(points[0]) == 1 ):
    #            print(points[0],points[5],h,w) #右眼坐标
    # =============================================================================
    #                frame[int(points[5]),int(points[0]) - 10 : int(points[0]) + 10] = [0,0,255]#画出右眼十字
    #                frame[int(points[5]) - 10:int(points[5]) + 10,int(points[0])] = [0,0,255] #画出右眼十字        
    #                frame[int(points[6]),int(points[1]) - 10:int(points[1]) + 10] = [255,0,0]#画出左眼十字
    #                frame[int(points[6])-10:int(points[6]) + 10,int(points[1])] = [255,0,0]#画出左眼十字
        # =============================================================================
                    Tresholdw = 1.8#宽的比例
                    Tresholdh = 0.17#高的比例
                    width_2 = int((points[1] - points[0])*Tresholdw/2)#眼列半距离
                    height_2 = int(Tresholdh*width_2)
                    mideyew = int((points[1] + points[0])/2) #眼列中点
                    mideyeh = int((points[5]+points[6])/2) #眼行中点
        # =============================================================================
    #    #             显示人眼区域
    #                frame[mideyeh-height_2,mideyew - width_2: mideyew + width_2] = [255,255,0]     
    #                frame[mideyeh+height_2,mideyew - width_2: mideyew + width_2] = [255,255,0] 
    #                frame[mideyeh-height_2:mideyeh+height_2,mideyew - width_2] = [255,255,0] 
    #                frame[mideyeh-height_2:mideyeh+height_2,mideyew + width_2] = [255,255,0] 
    #    #            显示人脸区域
                    frame[face_position[1],face_position[0]:face_position[2]] = [0,255,0]     
                    frame[face_position[3],face_position[0]:face_position[2]] = [0,255,0] 
                    frame[face_position[1]:face_position[3],face_position[0]] = [0,255,0]     
                    frame[face_position[1]:face_position[3],face_position[2]] = [0,255,0] 
                    cv2.imshow("Face",frame)
    # =============================================================================
    # =============================================================================
    #            旋转图像以保持水平
    #                theta = math.atan((points[1]-points[0])/(points[5]-points[6]))*60
    #                 if(theta>=0):
    #                     if(theta > 90): theta = 90
    #                     theta = theta -90
    #                 else:
    #                     if(theta < -90): theta = -90
    #                     theta = theta + 90
    #                 M = cv2.getRotationMatrix2D((mideyew/2,mideyeh/2),theta,0.6) 
    #                 dst = cv2.warpAffine(cropped,M,(cropped.shape[0],cropped.shape[1]))
    #                 cv2.imshow('Theta',dst)
    # ============================================================  =================
    # =============================================================================
                #    faceimg = frame[face_position[1]:face_position[3],face_position[0]:face_position[2]] 
                #    facesize = cv2.resize(faceimg, (180,225), interpolation=cv2.INTER_CUBIC ) #缩放为225*180
                #    cv2.imshow("Face", facesize)
                
    #                eyeimg = frame[mideyeh-height_2:mideyeh+height_2,mideyew - width_2:mideyew + width_2]
    #                eyesize = cv2.resize(eyeimg, (180,30), interpolation=cv2.INTER_CUBIC ) #缩放为30*180
    #                cv2.imshow("Eye", eyesize)
                    
                    L_eyeimg = frame[mideyeh-height_2:mideyeh+height_2,mideyew - int(5/6*width_2):mideyew - int(1/3 * width_2)]
                    R_eyeimg = frame[mideyeh-height_2:mideyeh+height_2,mideyew + int(1/3 * width_2):mideyew + int(5/6*width_2)]
                    
                    L_eyesize = cv2.resize(L_eyeimg, (30,20), interpolation=cv2.INTER_CUBIC ) #缩放为30*180
                    R_eyesize = cv2.resize(R_eyeimg, (30,20), interpolation=cv2.INTER_CUBIC ) #缩放为30*180
                    
                    L_eyegray = cv2.cvtColor(L_eyesize,cv2.COLOR_BGR2GRAY)
                    R_eyegray = cv2.cvtColor(R_eyesize,cv2.COLOR_BGR2GRAY)
                    
                    img_GuassianL = cv2.GaussianBlur(L_eyegray,(3,3),0)
                    trsL,_ = cv2.threshold(img_GuassianL, 20, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)  # Otsu 滤波
                    _,BlackimgL = cv2.threshold(img_GuassianL,5.5*trsL/7,255,cv2.THRESH_BINARY)
                    
                    img_GuassianR = cv2.GaussianBlur(R_eyegray,(3,3),0)
                    trsR,_ = cv2.threshold(img_GuassianR, 20, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)  # Otsu 滤波
                    _,BlackimgR = cv2.threshold(img_GuassianR,5.5*trsR/7,255,cv2.THRESH_BINARY)

                    
                    FillterimgL = Filtercol( BlackimgL,4)
                    begincolL,endcolL = Findboundcol(FillterimgL)
                    beginrowL,endrowL = Findboundrow(FillterimgL)
                    FillterimgL[:,begincolL] = 0
                    FillterimgL[:,endcolL] = 0#列
                    FillterimgL[beginrowL,:] = 0
                    FillterimgL[endrowL,:] = 0#行
                    
                    FillterimgR = Filtercol( BlackimgR,4)
                    begincolR,endcolR = Findboundcol(FillterimgR)
                    beginrowR,endrowR = Findboundrow(FillterimgR)
                    FillterimgR[:,begincolR] = 0
                    FillterimgR[:,endcolR] = 0#列
                    FillterimgR[beginrowR,:] = 0
                    FillterimgR[endrowR,:] = 0#行
                    
                   # print(begincol,beginrow)
                  #  L_eyesize[:,int((beginline+endline)/2)] = (0,255,255)
                  
                    L_eyesize = cv2.resize(L_eyesize, (300,200), interpolation=cv2.INTER_CUBIC ) #缩放
                    BlackimgL  = cv2.resize( BlackimgL, (300,200), interpolation=cv2.INTER_CUBIC ) #缩放
                    FillterimgL  = cv2.resize(FillterimgL, (300,200), interpolation=cv2.INTER_CUBIC ) #缩放
                    
                    R_eyesize = cv2.resize(R_eyesize, (300,200), interpolation=cv2.INTER_CUBIC ) #缩放
                    BlackimgR  = cv2.resize( BlackimgR, (300,200), interpolation=cv2.INTER_CUBIC ) #缩放
                    FillterimgR  = cv2.resize(FillterimgR, (300,200), interpolation=cv2.INTER_CUBIC ) #缩放
                    
                    
                    
                    ##画瞳孔
                    
                    if endcolL!= 0:
                       # radius = int((endcol - begincol) / 2) + 1
                       radiusL = 6
                       #print(radius)
                       FinalxL = int((begincolL + endcolL) / 2)
                       FinalyL = int((beginrowL + endrowL) / 2)
                       cv2.circle(L_eyesize,(FinalxL*10,FinalyL*10),2,(0,255,255),1,0,0)
                       cv2.circle(L_eyesize,(FinalxL*10,FinalyL*10),radiusL*10,(255,0,0),1,0,0)


                    
                     ##画瞳孔圆
                    if endcolR != 0:
                       # radius = int((endcol - begincol) / 2) + 1
                       radiusR = 6
                       #print(radius)
                       FinalxR = int((begincolR + endcolR) / 2)
                       FinalyR = int((beginrowR + endrowR) / 2)
                       cv2.circle(R_eyesize,(FinalxR*10,FinalyR*10),2,(0,255,255),1,0,0)
                       cv2.circle(R_eyesize,(FinalxR*10,FinalyR*10),radiusR*10,(255,0,0),1,0,0)
                

                       


                    #cv2.circle(L_eyesize,(Eyex*10,Eyey*10),5*10,(0,0,255),1,0,0)

                    cv2.imshow("REye", R_eyesize)
                    cv2.imshow("LEye", L_eyesize)

   #####发送数据       
            if(SuccessFlag == 1):
                SuccessFlag = 0
                if FinalxR<10:
                   msg = '0'
                msg = msg + str(FinalxR)
                if FinalyR<10:
                   msg = msg + '0'
                msg = msg + str(FinalyR)
            #else:
                #msg = '0000'
            connection.send(bytes(msg, encoding="ascii"))
                    
                    
                #print(bytes(msg, encoding='ascii'))
                    # cv2.imshow("BlackEye", Blackimg)
                    # cv2.imshow("Fillterimg", Fillterimg)
                    #cv2.imwrite('F:\\eye.jpg',L_eyeimg)
    # =============================================================================
                    #简单滤波
    #                eyegray = cv2.cvtColor(L_eyeimg,cv2.COLOR_BGR2GRAY)
    #                
    #                trs,th1 = cv2.threshold(eyegray, 20, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)  # Otsu 滤波
    #                
    #               # print(trs);
    #                _,th2 = cv2.threshold(eyegray,6*trs/7,255,cv2.THRESH_BINARY)
    #                cv2.imshow("Treshold",th1)
    #                cv2.imshow("Treshold2",th2)
    
    #                (h,w)=th1.shape #返回高和宽
    #                a = [0 for z in range(0, w)] 
    #                #记录每一列的波峰
    #                for j in range(0,w): #遍历一列 
    #                    for i in range(0,h):  #遍历一行
    #                        if  th1[i,j]==0:  #如果改点为黑点
    #                            a[j]+=1  		#该列的计数器加一计数
    #                            th1[i,j]=255  #记录完后将其变为白色           
    #                for j  in range(0,w):  #遍历每一列
    #                    for i in range((h-a[j]),h):  #从该列应该变黑的最顶部的点开始向最底部涂黑
    #                        th1[i,j]=0   #涂黑
    #                cv2.imshow('img',th1)  
    
    
    # =============================================================================
    #                 霍夫曼圆
    #                gray = cv2.cvtColor(eyesize,cv2.COLOR_BGR2GRAY)
    #                circles= cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,100,120,30,1,0)
    #                for circle in circles[0]:
    #                    #圆的基本信息
    #                    print(circle[2])
    #                    #坐标行列
    #                    x=int(circle[0])
    #                    y=int(circle[1])
    #                    #半径
    #                    r=int(circle[2])
    #                    #在原图用指定颜色标记出圆的位置
    #                    eyesize=cv2.circle(eyesize,(x,y),r,(0,0,255),-1)
    #                #显示新图像
    #                cv2.imshow('Huff',eyesize)
    
    # =============================================================================
    # =============================================================================
    #             #边缘检测
    #                gray = cv2.cvtColor(eyeimg,cv2.COLOR_BGR2GRAY)
    #                canny = cv2.Canny(gray, 240,250)
    #                canny = np.uint8(np.absolute(canny))
    #                #cv2.imshow('Canny', np.hstack([gray,canny]))
    #                cv2.imshow('Canny', canny)
    # =============================================================================
    # =============================================================================
    #             #角点
    #                gray = cv2.cvtColor(eyeimg,cv2.COLOR_BGR2GRAY)
    #                harris = np.float32(gray)
    #                 # 输入图像必须是 float32 ,最后一个参数在 0.04 到 0.05 之间
    #                dst = cv2.cornerHarris(harris,2,3,0.04)
    #                 #result is dilated for marking the corners, not important
    #                dst = cv2.dilate(dst,None)  
    #                 # Threshold for an optimal value, it may vary depending on the image.
    #                eyeimg[dst>0.*dst.max()]=[0,0,255] 
    #                cv2.imshow('Harris',eyeimg)
    # =============================================================================
                   
            #cv2.imshow('Canny', np.hstack([gray,canny]))
                    
                   
                   
               # print(cropped.shape)
    # =============================================================================
    #             if(cropped.shape[0] >0 and cropped.shape[1] >0 ):
    #                 #scaled = cv2.resize(cropped, (160, 160), interpolation=cv2.INTER_CUBIC )  #这里取和负样本一样大小
    #    
    #            
    #             #cv2.imwrite('E:/TensorCode/facenet-master-Mine/Faceshut/Face'+str(frame_count) + '.jpg', scaled)
    #             
    #                 #cv2.imshow("capture", scaled)
    #                 #cv2.imshow("capture", cropped)
    #                 cv2.imshow("Face", cropped)
    # =============================================================================
    except:
        continue
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break

#server.close()
video_capture.release()
cv2.destroyAllWindows()
print('已结束')


# =============================================================================
# #coding=utf-8
# import cv2 
# import time
#  
# if __name__ == '__main__':
#  
#     cv2.namedWindow("camera",1)
#     #开启ip摄像头
#     video="http://admin:admin@192.168.2.149:8081/"   #此处@后的ipv4 地址需要修改为自己的地址
#     capture =cv2.VideoCapture(video)
#  
#     num = 0;
#     while True:
#         success,img = capture.read()
#         cv2.imshow("camera",img)
#  
#     #按键处理，注意，焦点应当在摄像头窗口，不是在终端命令行窗口
#         key = cv2.waitKey(10) 
#  
#         if key == 27:
#         #esc键退出
#             print("esc break...")
#             break
#         if key == ord(' '):
#              #保存一张图像
#             num = num+1
#             filename = "frames_%s.jpg" % num
#             cv2.imwrite(filename,img)
#  
#  
#     capture.release()
#     cv2.destroyWindow("camera")
# 
# =============================================================================
