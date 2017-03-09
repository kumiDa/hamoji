
import cv2
import numpy as np
import math

np.set_printoptions(threshold=np.nan)
cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
file = cv2.VideoWriter('output.AVI', fourcc, 20.0, (848, 480))
fgbg = cv2.createBackgroundSubtractorMOG2()
ROIheight, ROIwidth = 300, 300
emoji = (cv2.imread('emojis/COOLDUDE.jpg',1),cv2.imread('emojis/CONFUSED.jpg',1),cv2.imread('emojis/LOL.jpg',1),cv2.imread('emojis/LOVE.jpg',1),cv2.imread('emojis/WINK.jpg',1))

# rerad and handle frames
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        file.write(frame)
        # selecting ROI boundaries
        cv2.rectangle(frame, (ROIwidth, ROIheight), (100, 100), (255, 0, 0), 0)

        # prep the frames
        frameCropd = frame[100:ROIwidth, 100:ROIheight]
        frameGrayScl = cv2.cvtColor(frameCropd, cv2.COLOR_BGR2GRAY)
        frameGaussBlurr = cv2.GaussianBlur(frameGrayScl, (35, 35), 0)
        ret, frameThreshd = cv2.threshold(frameGaussBlurr, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # find the contours
        img, contour, hier = cv2.findContours(frameThreshd, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cnt = max(contour, key=lambda x: cv2.contourArea(x))

        # generate bounding rectangle
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frameCropd, (x, y), (x + w, y + h), (0, 0, 255), 0)
        hull = cv2.convexHull(cnt)

        # draw the contours
        drawContourOutline = np.zeros(frameCropd.shape, np.uint8)
        cv2.drawContours(drawContourOutline, [cnt], -1, (0, 255, 0), 0)
        cv2.drawContours(drawContourOutline, [hull], 0, (0, 0, 255), 0)

        # refine the defined hull
        hull = cv2.convexHull(cnt, returnPoints=False)
        defects = cv2.convexityDefects(cnt, hull)
        defectCount = 0
        cv2.drawContours(frameCropd, contour, -1, (0, 255, 0), 3)

        # analyse the frame with respect to angle params
        for i in range(defects.shape[0]):
            startLen, endLen, farLen, dLen = defects[i, 0]
            start = tuple(cnt[startLen][0])
            end = tuple(cnt[endLen][0])
            far = tuple(cnt[farLen][0])
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57
            if angle <= 90:
                defectCount += 1
                cv2.circle(frameCropd, far, 2, [0, 120, 255], -1)  # marker
            cv2.line(frameCropd, start, end, [0, 255, 0], 2)  # marker

        # include target output text on the frame
        if defectCount == 1:
            cv2.putText(frame, "1", (180, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
            drawEmoji=emoji[0]
        elif defectCount == 2:
            cv2.putText(frame, "2", (180, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
            drawEmoji=emoji[1]
        elif defectCount == 3:
            cv2.putText(frame, "3", (180, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
            drawEmoji=emoji[2]
        elif defectCount == 4:
            cv2.putText(frame, "4", (180, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
            drawEmoji=emoji[3]
        else:
            cv2.putText(frame, "5", (180, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
            drawEmoji = emoji[4]

        # display frames
        drawEmoji=cv2.resize(drawEmoji,(230,230))#resize emoji picture
        cv2.imshow('emoji', drawEmoji)
        cv2.imshow('frame1', drawContourOutline)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('`'):#press '`' to release the window and the session eventually
            break
    else:
        break

# release object session
cap.release()
file.release()
cv2.destroyAllWindows()
