# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 21:47:59 2019

@author: Beta
"""

#腐蚀滤波num为滤波点数
def Filtercol(img,num):
    Filterimg = img
    begin = 0
    blacknum = 0
    jmppointBlack = 0
    for col in range(0,img.shape[1]):
        for row in range(0,img.shape[0] - 1):
            if img[row, col] == 255 and img[row + 1, col] == 0:	
                begin = row + 1
                blacknum += 1
                for j in range(begin + 1,img.shape[0] - 1):
                    if img[j + 1, col] == 0:
                        blacknum += 1
                    else:
                        break			
                if blacknum <= num and begin + blacknum < img.shape[0]:
                    for j in range(begin,begin + blacknum):
                        Filterimg[j,col] = 255
                    jmppointBlack += 1
                row = row + blacknum
                blacknum = 0
    return Filterimg
    		#else if (jmppoint != 0 || Blackimg(row, col) == 1) Array[col] = 255;
    	
def Findboundcol(img):
    beginline = 0
    endline = 0
    whitecount = 0
    for col in range(0,img.shape[1]):#列
        for row in range(0,img.shape[0] - 1):
            if beginline == 0 and img[row, col] == 0 and img[row + 1, col] == 0:	
                beginline = col 
                break
            elif beginline != 0 and img[row, col] == 255:
                whitecount += 1
                if  whitecount >= img.shape[0] - 2:
                   endline = col
                   break
            else:
                whitecount = 0
        if endline != 0:
            break;
    return beginline,endline	

def Findboundrow(img):
    beginline = 0
    endline = 0
    whitecount = 0
    for row in range(0,img.shape[0]):#行
        for col in range(0,img.shape[1] - 1):
            if beginline == 0 and img[row, col] == 0 and img[row, col + 1] == 0:	
                beginline = row
                break
            elif beginline != 0 and img[row, col] == 255:
                whitecount += 1
                if  whitecount >= img.shape[1] - 2:
                   endline = row
                   break
            else:
                whitecount = 0
        if endline != 0:
            break;
    return beginline,endline