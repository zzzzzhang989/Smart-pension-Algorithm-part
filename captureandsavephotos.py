# -*- coding: utf-8 -*-
'''
使用web摄像头 (USB摄像头)捕捉图像并保存
'''

import cv2
import time
 
cap = cv2.VideoCapture(0)
cap.set(0,640) # set Width (the first parameter is property_id)
cap.set(1,480) # set Height
time.sleep(2)


for i in range(100):# 拍100张图片就结束
    ret, img = cap.read()
    cv2.imshow('img', img)
    cv2.imwrite('images/99/%d.jpg' %(i), img)
    
	# Press 'ESC' for exiting video
    k = cv2.waitKey(100) & 0xff 
    if k == 27:
        break
 
cap.release()
cv2.destroyAllWindows()