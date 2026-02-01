# speech.py
# This file handles text-to-speech. 
# I’m using macOS “say” command because it is simple and works reliably.
# The idea is: whenever the program needs to speak something,
# it will add that message into a queue, and a background thread will read them one by one.

import subprocess
import threading
import queue

# Queue to store messages that need to be spoken
speech_queue = queue.Queue()

def _speech_worker():
    # This runs in the background and keeps checking if there is any text to speak.
    while True:
        text = speech_queue.get()
        try:
            # macOS built-in TTS command
            subprocess.run(["say", text])
        except Exception as e:
            print("TTS Error:", e)
        speech_queue.task_done()


# Start a background thread so the main app never gets stuck waiting for audio
threading.Thread(target=_speech_worker, daemon=True).start()


def speak_text(text):
    """
    Adds the text to the speech queue.
    The worker thread will speak it out.
    This way, messages play in order without blocking the UI.
    """
    speech_queue.put(text)

