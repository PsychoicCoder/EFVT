from fer import FER
import cv2
cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')) # depends on fourcc available camera
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# cap.set(cv2.CAP_PROP_FPS, 60)
detector = FER()
while(True):
    if (cap.isOpened()):
        ret, frame = cap.read()
        result = detector.detect_emotions(frame)
        if len(result)!=0:
            result=result[0]['emotions']
            cv2.rectangle(frame, (5 , int(480-(result['angry']*100))), (15, 480), (0, 0, 255), -1)
            cv2.rectangle(frame, (20, int(480-(result['disgust']*100))), (30, 480), (0, 255, 0), -1)
            cv2.rectangle(frame, (35, int(480-(result['fear']*100))), (45, 480), (255, 0, 255), -1)
            cv2.rectangle(frame, (50, int(480-(result['happy']*100))), (60, 480), (0, 255, 255), -1)
            cv2.rectangle(frame, (65, int(480-(result['sad']*100))), (75, 480), (255, 0, 0), -1)
            cv2.rectangle(frame, (80, int(480-(result['surprise']*100))), (90, 480), (255, 255, 0), -1)
            cv2.rectangle(frame, (95, int(480-(result['neutral']*100))), (105, 480), (100, 100, 100), -1)
            # cv2.putText(frame, 'angry', (10, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.FILLED)
            # cv2.putText(frame, 'disgust', (10, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.FILLED)
            # cv2.putText(frame, 'fear', (10, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.FILLED)
            # cv2.putText(frame, 'sad', (10, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.FILLED)
            # cv2.putText(frame, 'surprise', (10, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.FILLED)
            # cv2.putText(frame, 'neutral', (10, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.FILLED)
            cv2.imshow('Video', frame)
            # print("angry"+str(result['angry']*100))
            # print("disgust"+str(result['disgust']*100))
            # print("fear"+str(result['fear']*100))
            # print("happy"+str(result['happy']*100))
            # print("sad"+str(result['sad']*100))
            # print("surprise"+str(result['surprise']*100))
            # print("neutral"+str(result['neutral']*100))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break