import cv2
import numpy as np

np.set_printoptions(threshold=np.nan)
cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
file = cv2.VideoWriter('output.AVI',fourcc, 20.0, (848,480))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        file.write(frame)
        frameGrayScl=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frameGaussBlurr=cv2.GaussianBlur(frameGrayScl,(63,63),0)
        ret,frameThreshd=cv2.threshold(frameGaussBlurr,127,255,cv2.THRESH_BINARY+cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        # img,contours,heir=cv2.findContours(frameThreshd,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        # print heir ##debugging
        cv2.imshow('frame',frameThreshd)
        if cv2.waitKey(1) & 0xFF == ord('`'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
file.release()
cv2.destroyAllWindows()
