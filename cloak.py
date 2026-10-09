import cv2
import numpy as np
import time
import os
import urllib.request
import ssl

# ---------------------------------------------------------
# 0. Check & Download Required Files Automatically
# ---------------------------------------------------------
MASK_FILE = "spiderman_mask.png"
CASCADE_FILE = "haarcascade_frontalface_default.xml"
CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"

# Download Cascade XML if missing
if not os.path.exists(CASCADE_FILE):
    print("Downloading face detection model...")
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(CASCADE_URL, context=ctx) as response, open(CASCADE_FILE, 'wb') as out_file:
            out_file.write(response.read())
        print("Downloaded face model successfully!")
    except Exception as e:
        print(f"Failed to download face model: {e}")

# Generate default mask if missing
def generate_default_mask(filename):
    mask = np.zeros((400, 400, 4), dtype=np.uint8)
    cv2.ellipse(mask, (200, 200), (160, 190), 0, 0, 360, (0, 0, 220, 255), -1)
    
    pts_left = np.array([[110, 150], [180, 185], [130, 220]], np.int32)
    pts_right = np.array([[290, 150], [220, 185], [270, 220]], np.int32)
    
    cv2.fillPoly(mask, [pts_left], (255, 255, 255, 255))
    cv2.fillPoly(mask, [pts_right], (255, 255, 255, 255))
    
    cv2.polylines(mask, [pts_left], True, (0, 0, 0, 255), 8)
    cv2.polylines(mask, [pts_right], True, (0, 0, 0, 255), 8)
    
    cv2.line(mask, (200, 200), (200, 10), (0, 0, 0, 200), 3)
    cv2.line(mask, (200, 200), (40, 100), (0, 0, 0, 200), 3)
    cv2.line(mask, (200, 200), (360, 100), (0, 0, 0, 200), 3)
    
    cv2.imwrite(filename, mask)

if not os.path.exists(MASK_FILE):
    generate_default_mask(MASK_FILE)

# ---------------------------------------------------------
# 1. Initialize Models & Resources
# ---------------------------------------------------------
face_cascade = cv2.CascadeClassifier(CASCADE_FILE)

if face_cascade.empty():
    print("Error: Could not load face cascade XML file.")
    exit()

mask_img = cv2.imread(MASK_FILE, cv2.IMREAD_UNCHANGED)

def overlay_transparent(background, overlay, x, y, size=None):
    if overlay is None:
        return background
    
    if size:
        overlay = cv2.resize(overlay, size)

    h, w, _ = overlay.shape
    bg_h, bg_w, _ = background.shape

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bg_w, x + w), min(bg_h, y + h)

    overlay_x1 = max(0, -x)
    overlay_y1 = max(0, -y)
    overlay_x2 = overlay_x1 + (x2 - x1)
    overlay_y2 = overlay_y1 + (y2 - y1)

    if x2 <= x1 or y2 <= y1:
        return background

    overlay_crop = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
    
    if overlay_crop.shape[2] == 4:
        alpha_mask = overlay_crop[:, :, 3] / 255.0
        alpha_inv = 1.0 - alpha_mask

        for c in range(0, 3):
            background[y1:y2, x1:x2, c] = (
                alpha_mask * overlay_crop[:, :, c] + alpha_inv * background[y1:y2, x1:x2, c]
            )
    return background

# ---------------------------------------------------------
# 2. Webcam Setup & Background Capture
# ---------------------------------------------------------
cap = cv2.VideoCapture(0)
time.sleep(2)

print("Capturing background... Please move out of frame.")
for i in range(60):
    ret, background = cap.read()

if not ret:
    print("Failed to capture background. Exiting...")
    cap.release()
    cv2.destroyAllWindows()
    exit()

background = np.flip(background, axis=1)

# Fullscreen Display Window
cv2.namedWindow("Invisibility Cloak + Spider-Man", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Invisibility Cloak + Spider-Man", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("Background captured! Hold up your cloak.")

# ---------------------------------------------------------
# 3. Main Processing Loop
# ---------------------------------------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = np.flip(frame, axis=1)

    # --- A. Invisibility Cloak Effect ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    cloak_mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    cloak_mask = cv2.morphologyEx(cloak_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cloak_mask = cv2.dilate(cloak_mask, np.ones((3, 3), np.uint8), iterations=1)
    mask_inv = cv2.bitwise_not(cloak_mask)

    res1 = cv2.bitwise_and(background, background, mask=cloak_mask)
    res2 = cv2.bitwise_and(frame, frame, mask=mask_inv)
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    # --- B. Spider-Man Mask Detection & Overlay ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    if mask_img is not None:
        for (x, y, w, h) in faces:
            pad_w = int(w * 0.3)
            pad_h = int(h * 0.4)
            
            mask_x = x - pad_w // 2
            mask_y = y - pad_h // 2
            mask_w = w + pad_w
            mask_h = h + pad_h

            final_output = overlay_transparent(final_output, mask_img, mask_x, mask_y, (mask_w, mask_h))

    # --- C. Display Output ---
    cv2.imshow("Invisibility Cloak + Spider-Man", final_output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()