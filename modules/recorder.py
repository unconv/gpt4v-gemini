import audioop
import whisper
import pyaudio
import wave
import os

from config import config

whisper_model = whisper.load_model("base")
ambient_detected = False
speech_volume = 100

def transcribe(audio_file):
    result = whisper_model.transcribe(
        audio_file,
        fp16=False,
        no_speech_threshold=0.1,
        initial_prompt="mm-hmm, cough, tshh, pfft, swoosh"
    )

    return result["text"].strip()

def live_speech(wait_time=10, transcribe_audio=True, processing=None, ui_queue=None, winwidth=0, winheight=0):
    global ambient_detected
    global speech_volume

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024

    audio = pyaudio.PyAudio()

    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    frames = []
    recording = False
    frames_recorded = 0

    while True:
        frames_recorded += 1
        data = stream.read(CHUNK)
        rms = audioop.rms(data, 2)

        if not ambient_detected:
            if frames_recorded < 40:
                if frames_recorded == 1:
                    print("Detecting ambient noise...")
                if frames_recorded > 5:
                    if speech_volume < rms:
                        speech_volume = rms
                continue
            elif frames_recorded == 40:
                print("Listening...")
                speech_volume = speech_volume * config["speech_threshold"]
                ambient_detected = True

        if rms > speech_volume:
            if not recording:
                if processing:
                    if processing.value == True:
                        continue
                    with processing.get_lock():
                        processing.value = True
                if ui_queue:
                    # show "listening..."
                    ui_queue.put({
                        "type": "draw_text",
                        "args": {
                            "text": "listening...",
                            "position": (winwidth//2, int(winheight*0.9)),
                            "color": (150, 150, 150)
                        }
                    })
                print("Voice detected!")
            recording = True
            frames_recorded = 0
        elif recording and frames_recorded > wait_time:
            recording = False

            wf = wave.open("audio.wav", 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            if transcribe_audio:
                result = transcribe("audio.wav")

                os.remove("audio.wav")

                yield result
            else:
                yield "audio.wav"

            if ui_queue:
                # hide "listening..."
                ui_queue.put({
                    "type": "draw_text",
                    "args": {
                        "text": "listening...",
                        "position": (winwidth//2, int(winheight*0.9)),
                        "color": (0, 0, 0)
                    }
                })

            frames = []

        if recording:
            frames.append(data)

    # TODO: do these when breaking from generator
    stream.stop_stream()
    stream.close()
    audio.terminate()