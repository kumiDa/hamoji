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

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('`1'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
file.release()
cv2.destroyAllWindows()
