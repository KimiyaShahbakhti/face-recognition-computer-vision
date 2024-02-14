import cv2 as cv
video = cv.VideoCapture(0)
fgbg = cv.bgsegm.createBackgroundSubtractorMOG() #original image - bg
while True:
    ret, frame = video.read()
    if frame is None:
        break
    fgmask = fgbg.apply(frame)
    cv.imshow('Frame', frame)
    cv.imshow('bg sub', fgmask)
    keyboard = cv.waitKey(30)
    if keyboard == 'q':
        break
video.release()
cv.destroyAllWindows()