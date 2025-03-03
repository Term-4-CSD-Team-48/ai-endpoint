import requests
import cv2
import base64
import numpy as np
import json

# Define the URL of the FastAPI endpoint
url = "http://127.0.0.1:8080/invocations"


def send_frame_to_api(frame, prompt=None):
    # Encode the frame as a JPEG image
    _, buffer = cv2.imencode('.jpg', frame)
    frame_bytes = buffer.tobytes()

    # Prepare the files and data for the request
    files = {"file": ("frame.jpg", frame_bytes, "image/jpeg")}
    data = {"prompt": json.dumps(prompt)} if prompt else {}

    # Send the request to the API
    response = requests.post(url, files=files, data=data)
    return response.json()

# Set the mouse callback function for the window


def click_event(event, x, y, flags, param):
    global prompt
    global changed_points
    if event == cv2.EVENT_LBUTTONDOWN:
        prompt = {"x": x, "y": y}
        changed_points = True


prompt = {"x": 0, "y": 0}
changed_points = True
cv2.namedWindow("frame")
cv2.setMouseCallback("frame", click_event)


def main():
    global changed_points
    global prompt

    # Open a connection to the webcam
    cap = cv2.VideoCapture("IMG_4227.mov")
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,
                                           cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter("sam2.1.avi",
                          cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Send the frame to the API
        if changed_points:
            changed_points = False
            response = send_frame_to_api(frame, prompt)
            print("pROMPT SENT")
        else:
            response = send_frame_to_api(frame)
            print('NO SENT')

        # Decode the base64-encoded frame from the response
        frame_base64 = response["frame"]
        frame_data = base64.b64decode(frame_base64)
        frame = np.frombuffer(frame_data, dtype=np.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        # Display the frame
        cv2.imshow("frame", frame)

        # Check if the object is on screen
        object_on_screen = response["object_on_screen"]
        print(f"Object on screen: {object_on_screen}")

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        out.write(frame)

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()
    out.release()


if __name__ == "__main__":
    main()
