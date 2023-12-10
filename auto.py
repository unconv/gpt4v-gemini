from multiprocessing import Process, Queue, Value
import modules.cv2_stream as cv2_stream
from playsound import playsound
from openai import OpenAI
from queue import Empty
from PIL import Image
import base64
import shutil
import time
import os

import modules.recorder as recorder
import modules.helpers as helpers

client = OpenAI()

def motion_fn(queue: Queue, stream_url, processing: Value):
    for collage in cv2_stream.detect_changes(stream_url, processing=processing):
        print("Motion detected!")
        collage.save("collage.jpg", format="JPEG")

        queue.put({
            "image": collage
        })

        with processing.get_lock():
            processing.value = False

def voice_fn(queue: Queue, processing: Value):
    for audio_file in recorder.live_speech(60, transcribe_audio=False, processing=processing):
        if os.path.exists("collage.jpg"):
            image = Image.open("collage.jpg")
            os.remove("collage.jpg")
        else:
            shutil.copy("frame.jpg", "detect.jpg")
            try:
                image = Image.open("detect.jpg")
            except OSError:
                time.sleep(0.1)
                shutil.copy("frame.jpg", "detect.jpg")
                image = Image.open("detect.jpg")

            os.remove("detect.jpg")

        queue.put({
            "audio": audio_file,
            "image": image
        })

        with processing.get_lock():
            processing.value = False

def video_fn():
    for _ in cv2_stream.stream_frames(stream_url, "frame.jpg"):
        pass

def parse_message(message):
    image = helpers.image_b64(message["image"])

    with open("capture.jpg", "wb") as f:
        captured_image = base64.b64decode(image)
        f.write(captured_image)

    content = [
        {
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{image}"
        }
    ]

    if "audio" in message:
        text = recorder.transcribe(message["audio"])

        if helpers.filter_garbage(text):
            return []

        print("You: " + text)

        content.append({
            "type": "text",
            "text": "Message transcribed from recording (might contain mistakes): " + text,
        })
    else:
        if os.path.exists("collage.jpg"):
            os.remove("collage.jpg")

    return content

def empty_queue(queue):
    try:
        while True:
            queue.get_nowait()
    except Empty:
        pass

stream_url = helpers.get_stream()

if os.path.exists("collage.jpg"):
    os.remove("collage.jpg")

if os.path.exists("detect.jpg"):
    os.remove("detect.jpg")

if os.path.exists("frame.jpg"):
    os.remove("frame.jpg")

queue = Queue()

processing_audio = Value('b', False)
processing_video = Value('b', False)

print("Starting video process...")
video_process = Process(target=video_fn)
video_process.start()

print("Starting motion process...")
motion_process = Process(target=motion_fn, args=(queue, stream_url, processing_video))
motion_process.start()

print("Starting voice process...")
voice_process = Process(target=voice_fn, args=(queue, processing_audio), daemon=True)
voice_process.start()

messages = [
    {
        "role": "system",
        "content": """You are an AI assistant that reports what it sees in short sentences. You are connected to a video feed and will be shown a collage of frames from a video and you will tell what what is happening in the video as if you are watching. Don't let the user know that you are seeing a sequence of images. Pretend as if you are a human seeing what is happening live. You are talking directly to the user. Assume any activity is performed by the user, who you shall refer to as "You". "I see you drew an elephant" or "I see a dog" or "I see you put a TV remote on the table". Keep your answers very concise. If nothing particularly interesting happened since the previous image sequence, answer with the JSON {"status": "NO_CHANGE"}""".strip(),
    }
]

while True:
    message = queue.get()

    content = parse_message(message)

    time.sleep(0.5)

    if processing_audio.value == True:
        while processing_audio.value == True:
            time.sleep(0.1)

    if processing_video.value == True:
        while processing_video.value == True:
            time.sleep(0.1)

    try:
        message2 = queue.get(timeout=0.5)
        if message2:
            content += parse_message(message2)
    except Empty:
        pass

    messages.append({
        "role": "user",
        "content": content
    })

    response = client.chat.completions.create(
        messages=messages,
        model="gpt-4-vision-preview",
        max_tokens=1024
    )

    response_message = response.choices[0].message
    response_text = response_message.content

    if '{"status": "NO_CHANGE"}' in response_text:
        messages.pop()
        continue

    messages.append(response_message)

    audio = client.audio.speech.create(
        input=response_text,
        model="tts-1",
        voice="onyx",
    )

    audio.stream_to_file("audio.mp3")
    print("GPT: " + response_text)

    with processing_audio.get_lock():
        processing_audio.value = True

    playsound("audio.mp3")
    os.remove("audio.mp3")

    time.sleep(0.2)

    empty_queue(queue)

    with processing_audio.get_lock():
        processing_audio.value = False

    with processing_video.get_lock():
        processing_video.value = False

video_process.join()
motion_process.join()
voice_process.join()
