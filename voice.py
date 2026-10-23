import json
import queue
import threading
import time

import sounddevice as sd
from vosk import KaldiRecognizer, Model, SetLogLevel

# Things a lecturer might say at the end. Vosk writes what it hears in lowercase.
END_PHRASES = [
    "see you next week",
    "see you next time",
    "that's it for today",
    "that is it for today",
    "that's all for today",
    "that is all for today",
    "lecture is over",
    "lesson is over",
    "class is over",
    "we'll stop there",
    "we will stop there",
]

last_heard = "(loading speech model...)"   # latest words we heard, shown on screen
lecture_over = threading.Event()            # gets "set" when someone says an end phrase


def listen():
    """Listen to the microphone forever. Set lecture_over when an end phrase is heard."""
    global last_heard
    SetLogLevel(-1)                       # hide Vosk's long log messages
    model = Model(lang="en-us")           # first run downloads a ~40 MB English model
    samplerate = int(sd.query_devices(kind="input")["default_samplerate"])
    recognizer = KaldiRecognizer(model, samplerate)
    audio = queue.Queue()

    def callback(indata, frames, time_info, status):
        audio.put(bytes(indata))          # the mic hands us small chunks of sound

    with sd.RawInputStream(samplerate=samplerate, blocksize=8000, dtype="int16",
                           channels=1, callback=callback):
        last_heard = "(listening...)"
        while True:
            chunk = audio.get()
            if recognizer.AcceptWaveform(chunk):
                # A finished sentence: check it for end phrases
                text = json.loads(recognizer.Result())["text"]
                if any(phrase in text for phrase in END_PHRASES):
                    lecture_over.set()
            else:
                # Still talking: just show what we've got so far
                text = json.loads(recognizer.PartialResult())["partial"]
            if text:
                last_heard = text


def start():
    """Listen in the background, so the camera loop keeps running at the same time."""
    threading.Thread(target=listen, daemon=True).start()


# Test the microphone on its own with:  python voice.py
if __name__ == "__main__":
    start()
    print("Speak! Say 'see you next week' to finish.")
    shown = ""
    while not lecture_over.is_set():
        if last_heard != shown:
            print("Heard:", last_heard)
            shown = last_heard
        time.sleep(0.2)
    print("Heard:", last_heard)
    print("Lecture over detected!")