import cv2
import numpy as np

np.set_printoptions(threshold=np.nan)

vidO=cv2.VideoCapture(0)
fourCC=cv2.VideoWriter_fourcc(*'XVID')
vidOutput=cv2.VideoWriter('output.mkv',fourCC,60,(1920,1080))

while (vidO.isOpened()):
    ret,frame=vidO.read()
    if ret==True:
        vidOutput.write(frame)
        cv2.imshow('Video Captured',frame)
        if cv2.waitKey(1) & 0xFF==ord('`'):
            break

vidO.release()
vidOutput.release()
cv2.destroyAllWindows()

print bool(cv2.waitKey(0))
print bool(cv2.waitKey(1))

# cv2.VideoCapture(