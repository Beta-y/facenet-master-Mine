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
video_capture = cv2.VideoCapture(0)
#video_capture = cv2.VideoCapture("E:\\TensorCode\\facenet-master-Mine\\src\\Test.mp4")

totalimgnum = 0;
successnum = 0;

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

while True:
    ret, orimage = video_capture.read()
    totalimgnum += 1
    orimage = cv2.flip(orimage,1)#水平翻转
    frame = cv2.resize(orimage, (int(orimage.shape[1]/4),int(orimage.shape[0]/4)), interpolation=cv2.INTER_CUBIC ) #缩放为30*180
    try:
    #    cv2.imshow("原图", frame)
        h=frame.shape[0]
        w=frame.shape[1]
        #每1帧采集一张人脸，这里采样不进行灰度变换，直接保存彩色图
        if(capture_count%capture_interval == 0): 
            
            bounding_boxes, points = align.detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
            nrof_faces = bounding_boxes.shape[0]
    #        print(points.shape) #摄像头原图像尺寸
            
            for face_position in bounding_boxes: #因为只采集一张人脸，所以实际只遍历一次
                successnum += 1
                
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
                    orimage[4*face_position[1],4*face_position[0]:4*face_position[2]] = [0,255,0]    #上横线 
                    orimage[4*face_position[3],4*face_position[0]:4*face_position[2]] = [0,255,0] 
                    orimage[4*face_position[1]:4*face_position[3],4*face_position[0]] = [0,255,0]     
                    orimage[4*face_position[1]:4*face_position[3],4*face_position[2]] = [0,255,0] 
                   # cv2.imshow("Face",orimage)
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
    # =============================================================================
    # =============================================================================
#                    faceimg = frame[face_position[1]:face_position[3],face_position[0]:face_position[2]] 
#                    facesize = cv2.resize(faceimg, (180,225), interpolation=cv2.INTER_CUBIC ) #缩放为225*180
                    #bounding_boxes_plus, points_plus = align.detect_face.detect_face(facesize, minsize, pnet, rnet, onet, threshold, factor)
#                    for face_position_plus in bounding_boxes:
#                        face_position_plus=face_position_plus.astype(int)
#                        cropped_plus = facesize[face_position_plus[1]:face_position_plus[3],face_position_plus[0]:face_position_plus[2],:]
#                        if(len(points_plus) == 10 and len(points_plus[0]) == 1 ):
#                                facesize[int(points_plus[5]),int(points_plus[0]) - 10 : int(points_plus[0]) + 10] = [0,0,255]#画出右眼十字
#                                facesize[int(points_plus[5]) - 10:int(points_plus[5]) + 10,int(points_plus[0])] = [0,0,255] #画出右眼十字        
#                                facesize[int(points_plus[6]),int(points_plus[1]) - 10:int(points_plus[1]) + 10] = [255,0,0]#画出左眼十字
#                                facesize[int(points_plus[6])-10:int(points_plus[6]) + 10,int(points_plus[1])] = [255,0,0]#画出左眼十字
#                                cv2.imshow("Face_plus", facesize)
                            
                            
                            
                            
    #                eyeimg = frame[mideyeh-height_2:mideyeh+height_2,mideyew - width_2:mideyew + width_2]
    #                eyesize = cv2.resize(eyeimg, (180,30), interpolation=cv2.INTER_CUBIC ) #缩放为30*180
    #                cv2.imshow("Eye", eyesize)
                    
                   # L_eyeimg = frame[mideyeh-height_2:mideyeh+height_2,mideyew - int(5/6*width_2):mideyew - int(1/3 * width_2)]
                    R_eyeimg = orimage[4*(mideyeh-height_2):4*(mideyeh+height_2),4*(mideyew + int(1/3 * width_2)):4*(mideyew + int(5/6*width_2))]
                    
                    L_eyeimg = R_eyeimg
                    L_eyesize = cv2.resize(L_eyeimg, (30,20), interpolation=cv2.INTER_CUBIC ) #缩放为30*20
                    
                    L_eyegray = cv2.cvtColor(L_eyesize,cv2.COLOR_BGR2GRAY)
                    img_Guassian = cv2.GaussianBlur(L_eyegray,(3,3),0)
                    trs,_ = cv2.threshold(img_Guassian, 20, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)  # Otsu 滤波
                    _,Blackimg = cv2.threshold(img_Guassian,5.5*trs/7,255,cv2.THRESH_BINARY)
                    
                    Fillterimg = Filtercol( Blackimg,4)  #4个像素点为最小单位的纵向腐蚀滤波
                    begincol,endcol = Findboundcol(Fillterimg)#找列边界
                    beginrow,endrow = Findboundrow(Fillterimg)#找行边界
                    #画边界线
                    Fillterimg[:,begincol] = 0
                    Fillterimg[:,endcol] = 0#列
                    Fillterimg[beginrow,:] = 0
                    Fillterimg[endrow,:] = 0#行
                    
                   # print(begincol,beginrow)
                  #  L_eyesize[:,int((beginline+endline)/2)] = (0,255,255)
                    
                    L_eyesize = cv2.resize(L_eyesize, (300,200), interpolation=cv2.INTER_CUBIC ) #缩放
                    Blackimg  = cv2.resize( Blackimg, (300,200), interpolation=cv2.INTER_CUBIC ) #缩放
                    Fillterimg  = cv2.resize(Fillterimg, (300,200), interpolation=cv2.INTER_CUBIC ) #缩放
                    
                    ##画瞳孔
                    
                    
                     ##画瞳孔圆
                    if endcol != 0:
                      # radius = int((endcol - begincol) / 2) + 1
                       radius = 6
                       #print(radius)
                       Finalx = int((begincol + endcol) / 2)
                       Finaly = int((beginrow + endrow) / 2)
                       cv2.circle(L_eyesize,(Finalx*10,Finaly*10),2,(0,255,255),1,0,0)
                       cv2.circle(L_eyesize,(Finalx*10,Finaly*10),radius*10,(255,0,0),1,0,0)
                        
                    #cv2.circle(L_eyesize,(Eyex*10,Eyey*10),5*10,(0,0,255),1,0,0)
                    cv2.imshow("Eye", L_eyesize)
                    cv2.imshow("Origion", orimage)
#                    cv2.imshow("Eye", L_eyesize)
#                    cv2.imshow("BlackEye", Blackimg)
#                    cv2.imshow("Fillterimg", Fillterimg)       
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
print("总帧数:",totalimgnum)
print("识别人脸帧数:",successnum)
print("成功率:",successnum/totalimgnum)
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
