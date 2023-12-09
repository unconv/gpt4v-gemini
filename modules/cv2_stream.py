from PIL import Image
import numpy as np
import math
import cv2

def make_collage(frames, border=35):
    frame_count = len(frames)

    rows = 1
    while frame_count / rows > 4:
        rows += 1
    per_row = math.ceil(frame_count / rows)

    try:
        image1 = Image.fromarray(frames[0])
        collage = Image.new('RGB', (image1.width*per_row+border*(per_row-1), image1.height*rows+border*(per_row-1)))
        collage.paste(image1, (0, 0))
    except OSError:
        print("Error saving collage...")
        return

    pos_x = image1.width + border
    pos_y = 0

    for i, frame in enumerate(frames[1:]):
        try:
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

def detect_changes(stream_url, count=9, min_frames=5, big_movement_threshold=2, processing=None):
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
    previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    previous_frame_rgb = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2RGB)

    # Still frame counter
    still_frame_counter = 0

    # Motion counter
    motion_counter = 0
    motions = {"0": []}

    frame_counter = 0
    big_movement = 0
    while cap.isOpened():
        frame_counter += 1
        # Capture frame-by-frame
        ret, current_frame = cap.read()
        if not ret:
            break

        # Convert current frame to grayscale
        gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)

        # Calculate the absolute difference
        frame_diff = cv2.absdiff(previous_frame, gray_frame)

        # Threshold for significant change
        _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

        # Count the number of changed pixels
        change_count = np.sum(thresh != 0)

        # If significant change is detected, save the frame
        if change_count > 8_000 and motion_counter: # Threshold for change, adjust as needed
            if change_count > 80_000:
                big_movement += 1

            if processing:
                with processing.get_lock():
                    processing.value = True

            motions[str(motion_counter)].append(previous_frame_rgb)
            motions[str(motion_counter)].append(rgb_frame)
            still_frame_counter = 0
        else:
            if still_frame_counter < 2 and motion_counter:
                motions[str(motion_counter)].append(rgb_frame)
            still_frame_counter += 1

        if still_frame_counter == 40:
            frames = motions[str(motion_counter)]
            frame_count = len(frames)
            if frame_count > min_frames and big_movement > big_movement_threshold:
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
            motion_counter += 1
            motions[str(motion_counter)] = []
            big_movement = 0

        # Update the previous frame
        previous_frame = gray_frame.copy()
        previous_frame_rgb = rgb_frame.copy()

        # Display the frame (optional)
        #cv2.imshow('Frame', current_frame)

        # Press Q on keyboard to exit the loop
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

    # Release the video capture object
    cap.release()

    # Close all frames
    cv2.destroyAllWindows()

def stream_frames(stream_url, output_file):
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

        if frame_number % 10 == 0:
            # Save frame
            cv2.imwrite(output_file, frame)

    # Release the video capture object
    cap.release()

    # Close all frames
    cv2.destroyAllWindows()
