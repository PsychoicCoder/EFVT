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

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
# modelt = HuggingFaceModel.Text.Bert_Tiny2
# tr = TextRecognizer(model=modelt, device=device)
#
# modelv = HuggingFaceModel.Voice.WavLM
# vr = VoiceRecognizer(model=modelv, device=device)
#
# modelt = Model("vosk-model-small-ru-0.22") # полный путь к модели
# rec = KaldiRecognizer(modelt, 16000)

fr = FER()

frames = [2,3.2,'sdf']
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

def main():
    global frames
    print(frames)
    if len(frames) > 0:
        print(frames)
        frames = []
        print(frames)

main()