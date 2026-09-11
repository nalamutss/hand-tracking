import cv2
import numpy as np
import mediapipe as mp
from collections import deque

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

# Daftar filter "PORTAL"
FILTER_NAMES = ["GLITCH", "NEON", "SKETCH", "DUO-TONE", "PIXELATE", "INVERT"]


def filter_glitch(roi):
    """Geser channel warna (RGB shift) buat efek glitch."""
    h, w = roi.shape[:2]
    shift = max(3, w // 25)
    b, g, r = cv2.split(roi)
    b = np.roll(b, shift, axis=1)
    r = np.roll(r, -shift, axis=1)
    out = cv2.merge([b, g, r])
    # noise garis horizontal biar makin "glitchy"
    for _ in range(4):
        y = np.random.randint(0, h)
        th = np.random.randint(1, 4)
        out[y:y + th, :] = np.roll(out[y:y + th, :], np.random.randint(-15, 15), axis=1)
    return out


def filter_neon(roi):
    """Deteksi tepi lalu diwarnai neon (cyan)."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    edges = cv2.dilate(edges, None, iterations=1)
    out = np.zeros_like(roi)
    out[edges > 0] = (255, 255, 0)  # cyan (BGR)
    return out


def filter_sketch(roi):
    """Pencil sketch hitam-putih dari edge detection."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)


def filter_duotone(roi):
    """Threshold jadi dua warna (magenta & oranye)."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
    out = np.zeros_like(roi)
    out[mask == 255] = (180, 30, 255)   # magenta
    out[mask == 0] = (0, 165, 255)      # oranye
    return out


def filter_pixelate(roi):
    """Downscale lalu upscale biar pecah kotak-kotak."""
    h, w = roi.shape[:2]
    if h < 10 or w < 10:
        return roi
    small = cv2.resize(roi, (max(2, w // 12), max(2, h // 12)), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def filter_invert(roi):
    """Negatif warna."""
    return cv2.bitwise_not(roi)


FILTER_FUNCS = {
    "GLITCH": filter_glitch,
    "NEON": filter_neon,
    "SKETCH": filter_sketch,
    "DUO-TONE": filter_duotone,
    "PIXELATE": filter_pixelate,
    "INVERT": filter_invert,
}


def get_portal_rect(hand_landmarks_list, frame_w, frame_h):
    """
    Ambil titik jempol+telunjuk dari tiap tangan yang terdeteksi,
    lalu bikin bounding box (portal) dari titik-titik tersebut.
    Butuh minimal 2 tangan supaya portal muncul.
    """
    if len(hand_landmarks_list) < 2:
        return None

    pts = []
    for hand_landmarks in hand_landmarks_list:
        for idx in (4, 8):  # thumb tip, index tip
            lm = hand_landmarks.landmark[idx]
            pts.append((int(lm.x * frame_w), int(lm.y * frame_h)))

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1, x2 = max(0, min(xs)), min(frame_w, max(xs))
    y1, y2 = max(0, min(ys)), min(frame_h, max(ys))

    if (x2 - x1) < 30 or (y2 - y1) < 30:
        return None
    return x1, y1, x2, y2


def draw_hud(frame, filter_name):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (20, 20, 20), -1)
    cv2.putText(frame, f"PORTAL: {filter_name}", (12, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 180), 2)
    cv2.putText(frame, "[n] ganti filter   [q] keluar", (12, frame.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Tidak bisa membuka webcam.")
        return

    filter_idx = 0
    smooth_rect = deque(maxlen=5)  # smoothing biar portal ga jitter

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            rect = get_portal_rect(result.multi_hand_landmarks, w, h)
            if rect:
                smooth_rect.append(rect)

            if smooth_rect:
                arr = np.array(smooth_rect)
                x1, y1, x2, y2 = arr.mean(axis=0).astype(int)

                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    name = FILTER_NAMES[filter_idx]
                    processed = FILTER_FUNCS[name](roi)
                    frame[y1:y2, x1:x2] = processed
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 180), 2)
        else:
            smooth_rect.clear()

        draw_hud(frame, FILTER_NAMES[filter_idx])
        cv2.imshow("RETROLENS - Poke Python", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            filter_idx = (filter_idx + 1) % len(FILTER_NAMES)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()