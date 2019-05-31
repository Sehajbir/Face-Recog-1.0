import cv2,time, pickle
labels = {}
with open("labels.pickle", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}
recog = cv2.face.LBPHFaceRecognizer_create()
recog.read("model.yml")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
video = cv2.VideoCapture(0)
a = 1
while True:
    a = a+1
    check, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.15, minNeighbors=5)
    for x,y,w,h in faces:
        roi_gray = gray[y:y+h, x:x+w]
        img=cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
    id_, conf = recog.predict(roi_gray)
    if conf>=45:
        #print(id_)
        print(labels[id_])
        font = cv2.FONT_HERSHEY_SIMPLEX
        name = labels[id_]
        stroke = 2
        color = (0,255,0)
        cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
    cv2.imshow('Capturing', frame)
    key = cv2.waitKey(1)
    # cv2.destroyAllWindows()
    # break
    if key == 'q':
    	break
#print(a)
video.release()