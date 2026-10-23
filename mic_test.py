import numpy as np
import sounddevice as sd

MIC = None   # None = Windows default mic. Or put a number from the list it prints.

# Show every audio device, so you can see which mics exist
print(sd.query_devices())
mic = sd.query_devices(MIC, "input")
print("\nUsing mic:", mic["name"])
print("Speak! The bar should grow when you talk. (Ctrl + C to stop)\n")

samplerate = int(mic["default_samplerate"])


def callback(indata, frames, time_info, status):
    volume = np.sqrt(np.mean(indata ** 2))      # how loud this tiny piece of sound is
    bars = min(int(volume * 400), 50)
    print("\r[" + "#" * bars + " " * (50 - bars) + "]", end="", flush=True)


with sd.InputStream(device=MIC, channels=1, samplerate=samplerate,
                    blocksize=samplerate // 10, callback=callback):
    sd.sleep(20_000)   # listen for 20 seconds
print("\nDone")