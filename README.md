# GPT-4V Gemini

This is a crude demo project made to mimic the supposed [live video ingestion capabilities](https://www.youtube.com/watch?v=UIZAiXYceBI) of Google's multimodal Gemini LLM, but made with the GPT-4 Vision API.

Demo: https://youtu.be/UxQb88gENeg

## Setup

```shell
$ pip install -r requirements.txt
$ export OPENAI_API_KEY=YOUR_OPENAI_API_KEY
```

## Voice version (terminal)

To run the voice commanded terminal version, run the `voice.py` script.

```shell
$ python3 voice.py VIDEO_STREAM_URL
```

The assistant only reacts to voice commands.

## Motion version (terminal)

To run the motion detecting version, run the `motion.py` script.

```shell
$ python3 motion.py VIDEO_STREAM_URL
```

The assistants reacts every time motion is detected in the video. A tripod is recommended.

## Automatic version (terminal)

To run the automatic version that detects both voice commands and motion in the video, run the `auto.py` script.

```shell
$ python3 auto.py VIDEO_STREAM_URL
```

The assistants reacts every time motion is detected in the video or a voice command is given. A tripod is recommended.

## Automatic version with UI

There is also a version with a "UI" made with CV2 (it sucks but kinda works). It both listens to voice commands and detects motion in the video and automatically sends both to the GPT4V API.

```shell
$ python3 auto_with_ui.py VIDEO_STREAM_URL
```

## How to get a video stream URL

In my testing, I have used my phone camera as the video stream. For this, I used the [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam&pcampaignid=web_share) app on Play Store. I set the camera to 10 fps at 640x480 resolution.

The VIDEO_STREAM_URL is passed directly into `cv2.VideoCapture()`, so I guess you should be able to pass in a video file too, or any kind of video stream.

## Configuration

There is a `config.py` file where you can tweak some settings if you are having trouble with the motion detection or speech detection.

# Known issues

* GPT-4V API is often slow
* Sometimes the assistant response is detected as a user message
* The CV2 UI sucks and should be made with another way
* The CV2 UI can only be closed by hittin Ctrl+C in the terminal
