from cv2 import cv2 as cv

video = cv.VideoCapture(0)
fgbg = cv.bgsegm.createBackgroundSubtractorMOG()

while True:
    ret, frame = video.read()
    if frame is None:
        break
    fgmask = fgbg.apply(frame)
    cv.imshow('bg Sub', fgmask)
    filtered = cv.medianBlur(fgmask, ksize=5)  # remove noise
    cv.imshow('Filltered', filtered)
    canny = cv.Canny(filtered, 100, 200)
    cv.imshow('Canny', canny)
    keyboard = cv.waitKey(30)
    if keyboard == 'q':
        break

cap.release()
cv.destroyAllWindows()

