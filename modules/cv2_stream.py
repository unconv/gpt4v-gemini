from PIL import Image
import numpy as np
import math
import cv2

import modules.helpers as helpers
from config import config

def make_collage(frames, border=35):
    frame_count = len(frames)

    rows = 1
    while frame_count / rows > 4:
        rows += 1
    per_row = math.ceil(frame_count / rows)

    try:
        frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)
        image1 = Image.fromarray(frame)
        collage = Image.new('RGB', (image1.width*per_row+border*(per_row-1), image1.height*rows+border*(per_row-1)))
        collage.paste(image1, (0, 0))
    except OSError:
        print("Error saving collage...")
        return

    pos_x = image1.width + border
    pos_y = 0

    for i, frame in enumerate(frames[1:]):
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            collage.paste(image, (pos_x, pos_y))
        except OSError:
            print("Error adding changed frame...")
            continue

        pos_x += image.width + border

        if (i+2) % per_row == 0:
            pos_y += image.height + border
            pos_x = 0

    return collage

def detect_changes(stream_url, count=9, min_frames=5, max_frames=None, processing=None, frame_queue=None):
    if max_frames is None:
        max_frames = config["automatic_motion_cutoff"]

    # Create a VideoCapture object
    cap = cv2.VideoCapture(stream_url)

    # Check if the stream is opened successfully
    if not cap.isOpened():
        print("Error: Unable to open video stream")
        exit()

    # Read the first frame
    ret, previous_frame = cap.read()
    if not ret:
        print("Error: Unable to read video stream")
        cap.release()
        exit()

    # Convert the first frame to grayscale
    previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

    # Still frame counter
    still_frame_counter = 0

    # Frames
    frames = []

    frame_counter = 0
    big_movement = 0
    while cap.isOpened():
        frame_counter += 1
        # Capture frame-by-frame
        ret, current_frame = cap.read()
        if not ret:
            break

        if frame_queue:
            frame_queue.put(current_frame)

        # Convert current frame to grayscale
        gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Calculate the absolute difference
        frame_diff = cv2.absdiff(previous_frame_gray, gray_frame)

        # Threshold for significant change
        _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

        # Count the number of changed pixels
        change_count = np.sum(thresh != 0)

        # If significant change is detected, save the frame
        if change_count > current_frame.shape[1]*config["motion_threshold"]: # Threshold for change, adjust as needed
            if change_count > current_frame.shape[1]*config["big_motion_threshold"]:
                big_movement += 1

            if processing:
                with processing.get_lock():
                    processing.value = True

            frames.append(previous_frame)
            frames.append(current_frame)
            still_frame_counter = 0
        else:
            if still_frame_counter < 2:
                frames.append(current_frame)
            still_frame_counter += 1

        frame_count = len(frames)
        if still_frame_counter == config["still_frame_threshold"] or frame_count > max_frames:
            if frame_count > min_frames and big_movement >= int(frame_count/30):
                if frame_count > count:
                    sharp_frames = {}
                    frame_num = 0

                    while frame_count > 50:
                        frames = frames[0::2]
                        frame_count = len(frames)

                    for i, frame in enumerate(frames):
                        if str(frame_num) not in sharp_frames:
                            sharp_frames[str(frame_num)] = (0, None)

                        sharpness = helpers.sharpness(frame)
                        if sharp_frames[str(frame_num)][0] < sharpness:
                            sharp_frames[str(frame_num)] = (sharpness, frame)

                        if i % int(frame_count / count) == 0:
                            frame_num += 1
                    frames = []
                    for sharpness, frame in sharp_frames.values():
                        frames.append(frame)
                    frame_count = len(frames)
                step = int(frame_count / count)
                step = 1 if step < 1 else step
                spread_out_frames = list(reversed(frames[-1::-step]))[-count:] # i no gud at math

                # Yield new motion
                yield make_collage(spread_out_frames)

            if processing:
                with processing.get_lock():
                    processing.value = False

            frames = []
            big_movement = 0

        # Update the previous frame
        previous_frame_gray = gray_frame.copy()
        previous_frame = current_frame.copy()

        # Display the frame (optional)
        #cv2.imshow('Frame', current_frame)

        # Press Q on keyboard to exit the loop
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

    # Release the video capture object
    cap.release()

    # Close all frames
    cv2.destroyAllWindows()

def stream_frames(stream_url, output_file=None):
    # Create a VideoCapture object
    cap = cv2.VideoCapture(stream_url)

    # Check if the stream is opened successfully
    if not cap.isOpened():
        print("Error: Unable to open video stream")
        exit()

    frame_number = 0
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1

        if output_file:
            if frame_number % 10 == 0:
                # Save frame
                cv2.imwrite(output_file, frame)
        else:
            yield frame

    # Release the video capture object
    cap.release()

    # Close all frames
    cv2.destroyAllWindows()
