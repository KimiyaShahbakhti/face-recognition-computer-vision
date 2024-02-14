import cv2 as cv
video = cv.VideoCapture(0)
fgbg = cv.bgsegm.createBackgroundSubtractorMOG()
while True:
    ret, frame = video.read()
    if frame is None:
        break
    fgmask = fgbg.apply(frame)
    cv.imshow('bg Sub', fgmask)
    filtered = cv.medianBlur(fgmask,ksize=5) #remove noise
    cv.imshow('Filltered', filtered)
    keyboard = cv.waitKey(30)
    if keyboard == 'q':
        break
video.release()
cv.destroyAllWindows()