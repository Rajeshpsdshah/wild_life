#######DETECTION#########

import cv2
camera_port = 0
#camera = cv2.VideoCapture(camera_port)
cam = cv2.VideoCapture('http://192.168.5.125:8081/')
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

i = 0
while (True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # saving the captured face in the dataset folder
        cv2.imwrite("./dataSet/User." + str(i) + ".jpg", gray[y:y + h, x:x + w])

    cv2.imshow('feed', img)
    # wait for 100 miliseconds
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    i += 1
cam.release()
cv2.destroyAllWindows()
