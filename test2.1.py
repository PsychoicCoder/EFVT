import asyncio
import time

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
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')) # depends on fourcc available camera
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# cap.set(cv2.CAP_PROP_FPS, 60)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

modelt = HuggingFaceModel.Text.Bert_Tiny2
tr = TextRecognizer(model=modelt, device=device)

modelv = HuggingFaceModel.Voice.WavLM
vr = VoiceRecognizer(model=modelv, device=device)

modelt = Model("vosk-model-small-ru-0.22") # полный путь к модели
rec = KaldiRecognizer(modelt, 16000)

fr = FER()

frames = []
RATE = 16000*5
CHUNK = 4000
WAVE_OUTPUT_FILENAME = "output.wav"

p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=16000,
    input=True,
    frames_per_buffer=8000,
    input_device_index=4
)
stream.start_stream()

print("Готов")
async def text(data):
    if (rec.AcceptWaveform(data)) and (len(data) > 0):
        # answer = json.loads(rec.Result())
        print(rec.Result())
    else:
        if (len(data) > 0):
            print(rec.PartialResult() + tr.recognize(rec.PartialResult(), return_single_label=True))
            # tr.recognize('rec.PartialResult()', return_single_label=True)
async def voice(data,frames):
    frames.append(data)
    if len(frames) >= RATE / CHUNK:  # Save every second
        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'w')
        waveFile.setnchannels(1)
        waveFile.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        waveFile.setframerate(16000)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()
        frames = []

        print("v " + vr.recognize(WAVE_OUTPUT_FILENAME, return_single_label=True))

async def video():
    if (cap.isOpened()):
        ret, frame = cap.read()
        result = fr.detect_emotions(frame)
        if len(result) != 0:
            result = result[0]['emotions']
            cv2.rectangle(frame, (5, int(480 - (result['angry'] * 100))), (15, 480), (0, 0, 255), -1)
            cv2.rectangle(frame, (20, int(480 - (result['disgust'] * 100))), (30, 480), (0, 255, 0), -1)
            cv2.rectangle(frame, (35, int(480 - (result['fear'] * 100))), (45, 480), (255, 0, 255), -1)
            cv2.rectangle(frame, (50, int(480 - (result['happy'] * 100))), (60, 480), (0, 255, 255), -1)
            cv2.rectangle(frame, (65, int(480 - (result['sad'] * 100))), (75, 480), (255, 0, 0), -1)
            cv2.rectangle(frame, (80, int(480 - (result['surprise'] * 100))), (90, 480), (255, 255, 0), -1)
            cv2.rectangle(frame, (95, int(480 - (result['neutral'] * 100))), (105, 480), (100, 100, 100), -1)
            cv2.imshow('Video', frame)
            cv2.waitKey(1)
async def main():
    data = stream.read(4000, exception_on_overflow=False)

    await asyncio.gather(text(data), voice(data, frames), video())
    stream.stop_stream()
    stream.close()
    p.terminate()

start = time.time()
asyncio.run(main())
print(time.time() - start)