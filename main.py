import cv2 as cv  # type: ignore
import threading
import time
import winsound 

# Load haar cascades
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_alt.xml')
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

# Eye closure timer variables
eyes_closed_start_time = None
CLOSED_DURATION_THRESHOLD = 3  # Minimum time before beeping
beep_playing = False  

cap = cv.VideoCapture(0)
print("Press 'x' to exit")

# Thread function to play beep sound. This prevents a FPS drop.
def play_beep():
    global beep_playing
    try:
        winsound.Beep(10000, 2000)  # Beeps at 10000Hz for 2s
    except Exception as e:
        print(f"Error playing beep: {e}")
    finally:
        beep_playing = False  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.flip(frame, 1)

    # Grey scale conversion
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(50, 50))

    eyes_detected = False  

    for (fx, fy, fw, fh) in faces:
        # Define ROI for eyes within the face region
        roi_gray = gray[fy:fy + fh, fx:fx + fw]
        roi_color = frame[fy:fy + fh, fx:fx + fw]

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

        # If eyes are detected, they are highlighted with a ellipse
        if len(eyes) > 0:
            eyes_detected = True
            for (ex, ey, ew, eh) in eyes:
                center = (ex + ew // 2, ey + eh // 2)
                axes = (ew // 2, eh // 2)
                cv.ellipse(roi_color, center, axes, 0, 0, 360, (0, 255, 0), 2)

    # Check for eye closure
    if not eyes_detected:
        if eyes_closed_start_time is None:
            # Start timer when eyes are first not detected
            eyes_closed_start_time = time.time()
        else:
            # Check elapsed time
            elapsed_time = time.time() - eyes_closed_start_time
            if elapsed_time > CLOSED_DURATION_THRESHOLD:
                # Eyes are closed for too long - show warning and beep
                cv.putText(frame, "Get back to work!", (200, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if not beep_playing:
                    beep_playing = True
                    threading.Thread(target=play_beep, daemon=True).start()
    else:
        # Reset timer if eyes are detected
        eyes_closed_start_time = None
        beep_playing = False  # Reset flag when eyes are detected

    # Display the frame
    cv.imshow('Procrastionation detector', frame)

    # Break the loop on 'x' key press
    if cv.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv.destroyAllWindows()
