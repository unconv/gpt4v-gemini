from io import BytesIO
import base64
import re

def filter_garbage(message):
    if re.sub(r"[^a-z0-9]", "", message) == "":
        return True

    if message.count(",") / len(message) > 0.1:
        return True

    if message.strip().strip(",!?") in ["mm-hmm,", "cough,", "tshh,", "pfft,", "swoosh,"]:
        return True

    for word in ["mm-hmm,", "cough,", "tshh,", "pfft,", "swoosh,"]:
        if word in message:
            return True

    return False

def image_b64(image):
    if isinstance(image, str):
        with open(image, "rb") as f:
            return base64.b64encode(f.read()).decode()
    elif isinstance(image, bytes):
        return base64.b64encode(image).decode()
    else:
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode()
