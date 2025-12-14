import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import queue

# Load Faster-Whisper model
# -------------------------------
model_path = r"_models\faster-whisper-medium"
model = WhisperModel(model_path, device="cuda", compute_type="int8")

# Audio settings
# -------------------------------
samplerate = 16000
blocksize = 2048
buffer_length_sec = 4      # 4-second rolling buffer
buffer_slide_sec = 2       # slide buffer by 2 seconds
silence_threshold = 0.01   # skip very quiet audio

# Queue for audio chunks
# -------------------------------
audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    audio = indata.mean(axis=1)  # stereo -> mono
    #audio = audio / (np.max(np.abs(audio)) + 1e-7)  # normalize
    max_amp = np.max(np.abs(audio))
    if max_amp > 0.02:
        audio = audio / max_amp
    audio_queue.put(audio)

# Detect your Stereo Mix device
# -------------------------------
print("Available audio devices:")
for i, dev in enumerate(sd.query_devices()):
    print(i, dev['name'], "Input channels:", dev['max_input_channels'])

device_id = int(input("Enter Stereo Mix device ID: "))

# Start audio stream
# -------------------------------
stream = sd.InputStream(
    channels=2,
    samplerate=samplerate,
    device=device_id,
    blocksize=blocksize,
    callback=audio_callback
)
stream.start()
print("Streaming system audio... Press Ctrl+C to stop")

# Live transcription loop
# -------------------------------
buffer = np.zeros(0, dtype=np.float32)

try:
    while True:
        # Add new audio chunk
        chunk = audio_queue.get()
        buffer = np.concatenate((buffer, chunk))

        # Only process if buffer has enough audio
        if len(buffer) >= samplerate * buffer_length_sec:
            # Skip mostly silent audio
            if np.mean(np.abs(buffer)) < silence_threshold:
                buffer = buffer[int(samplerate * buffer_slide_sec):]  # slide buffer
                continue

            # Transcribe current buffer (Translate to english automatically)
            # To optimize use word_timestamps=False, beam_size=1
            #Default --> segments, _ = model.transcribe(buffer, language="en")
            segments, _ = model.transcribe(buffer, language=None, task="translate", word_timestamps=False, beam_size=1)
            for segment in segments:
                text = segment.text.strip()
                if text:
                    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {text}")

            # Slide the buffer forward for live streaming
            buffer = buffer[int(samplerate * buffer_slide_sec):]

except KeyboardInterrupt:
    print("Stopping transcription...")
    stream.stop()
    stream.close()
