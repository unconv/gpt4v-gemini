import modules.cv2_stream as cv2_stream
from multiprocessing import Process
from playsound import playsound
from openai import OpenAI
import shutil
import os

import modules.recorder as recorder
import modules.helpers as helpers

client = OpenAI()

stream_url = helpers.get_stream()

messages = [
    {
        "role": "system",
        "content": """You are an AI assistant that can see. The photos provided to you are the view from your eyes. Answer the user based on what you see. The user is holding the camera. If you see them pointing to something and asking what it is, tell them what it is. Don't say what you're looking at is an image, unless the image sent to you is of a physical image. Answer in short, concise answers.""",
    }
]

def write_changes():
    for _ in cv2_stream.stream_frames(stream_url, "frame.jpg"):
        pass

video_process = Process(target=write_changes)
video_process.start()

while True:
    for message in recorder.live_speech(60):
        if helpers.filter_garbage(message):
            break

        print("You: " + message)

        shutil.copy("frame.jpg", "detect.jpg")

        try:
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{helpers.image_b64('detect.jpg')}",
                    },
                    {
                        "type": "text",
                        "text": "Message transcribed from recording (might contain mistakes): " + message,
                    }
                ]
            })

            response = client.chat.completions.create(
                messages=messages,
                model="gpt-4-vision-preview",
                max_tokens=1024
            )

            response_message = response.choices[0].message
            response_text = response_message.content

            messages.append(response_message)
        except Exception as e:
            print(str(e))
            response_text = "Sorry, I missed that"
            messages.append({
                "role": "system",
                "content": "The user sent an invalid message"
            })
            messages.append({
                "role": "assistant",
                "content": response_text
            })

        audio = client.audio.speech.create(
            input=response_text,
            model="tts-1",
            voice="onyx",
        )

        audio.stream_to_file("audio.mp3")
        print("GPT: " + response_text)
        playsound("audio.mp3")
        os.remove("audio.mp3")

        break

video_process.join() # i really wanna join, but I can't
