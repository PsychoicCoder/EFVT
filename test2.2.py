import time
from threading import Thread

import pyaudio
import wave
import torch
import cv2
from vosk import Model, KaldiRecognizer
from aniemore.recognizers.voice import VoiceRecognizer
from aniemore.recognizers.text import TextRecognizer
from aniemore.models import HuggingFaceModel
from fer import FER

cap = cv2.VideoCapture(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

modelt = HuggingFaceModel.Text.Bert_Tiny2
tr = TextRecognizer(model=modelt, device=device)

modelv = HuggingFaceModel.Voice.WavLM
vr = VoiceRecognizer(model=modelv, device=device)

modelt = Model("vosk-model-small-ru-0.22") # полный путь к модели
rec = KaldiRecognizer(modelt, 16000)

fr = FER()
frame = {}
frames = []
RATE = 16000*1
CHUNK = 4000
WAVE_OUTPUT_FILENAME = "output.wav"
emotions={"neutral":0,"anger":1,"happiness":2,"sadness":3,"fear":4,"disgust":5,"surprise":6,"enthusiasm":7} # disgust-отвращение surprise-сюрприз
emotionsTe = -1
emotionsVo = -1
emotionsVi = -1
emotion = []
p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=16000,
    input=True,
    frames_per_buffer=8000,
    input_device_index=int(input("Номер устройства"))
)
stream.start_stream()

print("Готов")
def text(data):
    global emotionsTe, emotions
    if (rec.AcceptWaveform(data)) and (len(data) > 0):
        rec.Result()
        pass
    elif (len(data) > 0):
        emotionsTe = -1
        # emotionsTe = rec.PartialResult() + str(emotions.get(tr.recognize(rec.PartialResult(), return_single_label=True))) + tr.recognize(rec.PartialResult(), return_single_label=True)
        emotionsTe = emotions.get(tr.recognize(rec.PartialResult(), return_single_label=True))
def voice(date):
    global frames, emotionsVo, emotions
    frames.append(date)
    if len(frames) >= RATE / CHUNK:
        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))
        wf.close()
        frames = []
        emotionsVo = -1
        emotionsVo = emotions.get(vr.recognize(WAVE_OUTPUT_FILENAME, return_single_label=True))


def video(data):
    global frame
    if (cap.isOpened()):
        ret, frame = cap.read()
        result = fr.detect_emotions(frame)
        if len(result) != 0:
            result = result[0]['emotions']
            emotionsv = [result['neutral'] * 100,result['angry'] * 100,result['happy'] * 100,result['sad'] * 100,result['fear'] * 100,result['disgust'] * 100,result['surprise'] * 100]

            emotionsVi = emotionsv.index(max(emotionsv))
            cv2.rectangle(frame, (95, int(480 - (result['neutral'] * 100))), (105, 480), (100, 100, 100), -1)
            cv2.rectangle(frame, (5, int(480 - (result['angry'] * 100))), (15, 480), (0, 0, 255), -1)
            cv2.rectangle(frame, (50, int(480 - (result['happy'] * 100))), (60, 480), (0, 255, 255), -1)
            cv2.rectangle(frame, (65, int(480 - (result['sad'] * 100))), (75, 480), (255, 0, 0), -1)
            cv2.rectangle(frame, (35, int(480 - (result['fear'] * 100))), (45, 480), (255, 0, 255), -1)
            cv2.rectangle(frame, (20, int(480 - (result['disgust'] * 100))), (30, 480), (0, 255, 0), -1)
            cv2.rectangle(frame, (80, int(480 - (result['surprise'] * 100))), (90, 480), (255, 255, 0), -1)
def main():
    global emotion
    while True:
        data = stream.read(4000, exception_on_overflow=False)
        threads = [Thread(target=text, args=(data,), daemon=True), Thread(target=voice, args=(data,), daemon=True), Thread(target=video, args=(data,), daemon=True)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        print("Text:" + str(emotionsTe))
        print("Voice:" + str(emotionsVo))
        cv2.imshow('Video', frame)
        cv2.waitKey(1)
        if len(emotion) == 30:
            del emotion[0:2]
        emotion.append(emotionsTe)
        emotion.append(emotionsVo)
        emotion.append(emotionsVi)
        duplicates = [emotion.count(num) for num in set(emotion)]
        emotion_id = duplicates.index(max(duplicates))
        print(emotion_id)
    
start = time.time()
main()
print(time.time() - start)
stream.stop_stream()
stream.close()
p.terminate()