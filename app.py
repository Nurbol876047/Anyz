from __future__ import annotations

import atexit
import threading
import time
import urllib.request
from pathlib import Path
from typing import Generator

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, jsonify, send_file
from mediapipe.tasks.python import BaseOptions, vision


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

TRACKS = {
    "one": {
        "title": "1 саусақ: басқа әуен",
        "path": Path("/home/nurbol/Rendix/public/kaze-no-kata.mp3"),
    },
    "two": {
        "title": "2 саусақ: Төлеген Момбеков - Бозінген (күй)",
        "path": Path("/home/nurbol/Загрузки/Төлеген Момбеков - Бозінген (күй).mp3"),
    },
}

VALID_GESTURES = {1, 2}
HOLD_SECONDS = 0.85
REARM_SECONDS = 0.65
MILK_DELAY_SECONDS = 3.0


class GestureEngine:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.latest_jpeg: bytes = self._build_placeholder(
            "Camera standby", "Python + MediaPipe are starting..."
        )

        self.state = {
            "camera_ready": False,
            "current_gesture": 0,
            "gesture_label": "Қол көрінбей тұр",
            "candidate_gesture": 0,
            "hold_progress": 0.0,
            "event_id": 0,
            "active_track": None,
            "scene": "idle",
            "milk_visible": False,
            "milk_countdown": 0.0,
            "status": "Камераны іске қосу жүріп жатыр...",
            "tracks": {
                key: {
                    "title": meta["title"],
                    "available": meta["path"].exists(),
                }
                for key, meta in TRACKS.items()
            },
        }

        self._candidate_gesture = 0
        self._candidate_since = time.monotonic()
        self._last_trigger_gesture: int | None = None
        self._neutral_since = time.monotonic()
        self._pending_milk_at: float | None = None

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "camera_ready": self.state["camera_ready"],
                "current_gesture": self.state["current_gesture"],
                "gesture_label": self.state["gesture_label"],
                "candidate_gesture": self.state["candidate_gesture"],
                "hold_progress": self.state["hold_progress"],
                "event_id": self.state["event_id"],
                "active_track": self.state["active_track"],
                "scene": self.state["scene"],
                "milk_visible": self.state["milk_visible"],
                "milk_countdown": self.state["milk_countdown"],
                "status": self.state["status"],
                "tracks": self.state["tracks"],
            }

    def mjpeg_stream(self) -> Generator[bytes, None, None]:
        while not self.stop_event.is_set():
            with self.lock:
                frame = self.latest_jpeg
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
            time.sleep(0.05)

    def _run(self) -> None:
        try:
            model_path = self._ensure_model()
            landmarker_options = vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.65,
                min_hand_presence_confidence=0.6,
                min_tracking_confidence=0.6,
            )
            hand_landmarker = vision.HandLandmarker.create_from_options(
                landmarker_options
            )
        except Exception as exc:
            placeholder = self._build_placeholder(
                "MediaPipe model error",
                str(exc),
            )
            with self.lock:
                self.latest_jpeg = placeholder
                self.state["camera_ready"] = False
                self.state["status"] = f"MediaPipe жүктеу қатесі: {exc}"
            return

        cap: cv2.VideoCapture | None = None

        try:
            while not self.stop_event.is_set():
                if cap is None or not cap.isOpened():
                    cap = cv2.VideoCapture(0)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
                    time.sleep(0.2)

                if not cap.isOpened():
                    self._set_camera_missing()
                    time.sleep(1.0)
                    continue

                ok, frame = cap.read()
                if not ok:
                    self._set_camera_missing()
                    time.sleep(0.4)
                    continue

                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                try:
                    results = hand_landmarker.detect_for_video(
                        mp_image, int(time.monotonic() * 1000)
                    )
                except RuntimeError:
                    if self.stop_event.is_set():
                        break
                    raise

                gesture, label = self._detect_gesture(results)
                now = time.monotonic()
                self._advance_state(now, gesture, label)

                if results.hand_landmarks:
                    for hand_landmarks in results.hand_landmarks:
                        self._draw_hand(frame, hand_landmarks)

                self._annotate_frame(frame, gesture, label)
                encoded = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 82],
                )[1]

                with self.lock:
                    self.latest_jpeg = encoded.tobytes()
                    self.state["camera_ready"] = True

        finally:
            if cap is not None and cap.isOpened():
                cap.release()
            try:
                hand_landmarker.close()
            except RuntimeError:
                pass

    def _set_camera_missing(self) -> None:
        placeholder = self._build_placeholder(
            "Camera not found",
            "Check whether your webcam is available.",
        )
        with self.lock:
            self.latest_jpeg = placeholder
            self.state["camera_ready"] = False
            self.state["status"] = "Камера ашылмады. Құрылғыны тексеріңіз."

    def _advance_state(self, now: float, gesture: int, label: str) -> None:
        with self.lock:
            self.state["current_gesture"] = gesture
            self.state["gesture_label"] = label

            if gesture in VALID_GESTURES:
                self._neutral_since = now

                if gesture != self._candidate_gesture:
                    self._candidate_gesture = gesture
                    self._candidate_since = now

                hold_progress = min(1.0, (now - self._candidate_since) / HOLD_SECONDS)
                self.state["candidate_gesture"] = self._candidate_gesture
                self.state["hold_progress"] = hold_progress

                should_trigger = (
                    hold_progress >= 1.0 and self._last_trigger_gesture != gesture
                )
                if should_trigger:
                    self._trigger_gesture(now, gesture)
            else:
                self._candidate_gesture = 0
                self.state["candidate_gesture"] = 0
                self.state["hold_progress"] = 0.0

                if now - self._neutral_since >= REARM_SECONDS:
                    self._last_trigger_gesture = None

                if self.state["active_track"] is None and self._pending_milk_at is None:
                    self.state["status"] = (
                        "Камера жұмыс істеп тұр. Енді 1 немесе 2 саусақ көрсетіңіз."
                    )

            if self._pending_milk_at is not None:
                remaining = self._pending_milk_at - now
                if remaining <= 0:
                    self._pending_milk_at = None
                    self.state["milk_visible"] = True
                    self.state["milk_countdown"] = 0.0
                    self.state["scene"] = "milk"
                    self.state["status"] = (
                        "2 саусақ расталды. 3 секундтан кейін нар сүт берді."
                    )
                else:
                    self.state["milk_countdown"] = round(remaining, 1)
                    if self.state["scene"] == "two_wait":
                        self.state["status"] = (
                            f"Жаңыл доюға дайын. Сүтке дейін: {remaining:.1f} с"
                        )
            else:
                self.state["milk_countdown"] = 0.0

    def _trigger_gesture(self, now: float, gesture: int) -> None:
        self._last_trigger_gesture = gesture
        self.state["event_id"] += 1

        if gesture == 1:
            self._pending_milk_at = None
            self.state["active_track"] = "one"
            self.state["scene"] = "one"
            self.state["milk_visible"] = False
            self.state["status"] = "1 саусақ танылды. Бірінші әуен іске қосылды."
            return

        self._pending_milk_at = now + MILK_DELAY_SECONDS
        self.state["active_track"] = "two"
        self.state["scene"] = "two_wait"
        self.state["milk_visible"] = False
        self.state["status"] = "2 саусақ танылды. Дұрыс күй ойнап жатыр."

    def _detect_gesture(self, results) -> tuple[int, str]:
        if not results.hand_landmarks:
            return 0, "Қол көрінбей тұр"

        landmarks = results.hand_landmarks[0]

        def raised(tip_idx: int, pip_idx: int, mcp_idx: int) -> bool:
            tip = landmarks[tip_idx]
            pip = landmarks[pip_idx]
            mcp = landmarks[mcp_idx]
            return tip.y < pip.y - 0.02 and pip.y < mcp.y

        index_up = raised(8, 6, 5)
        middle_up = raised(12, 10, 9)
        ring_up = raised(16, 14, 13)
        pinky_up = raised(20, 18, 17)

        if index_up and not middle_up and not ring_up and not pinky_up:
            return 1, "1 саусақ танылды"
        if index_up and middle_up and not ring_up and not pinky_up:
            return 2, "2 саусақ танылды"
        return 0, "Тек 1 немесе 2 саусақ көрсетіңіз"

    def _draw_hand(self, frame: np.ndarray, hand_landmarks) -> None:
        height, width, _ = frame.shape
        connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

        for connection in connections:
            start = hand_landmarks[connection.start]
            end = hand_landmarks[connection.end]
            start_point = (int(start.x * width), int(start.y * height))
            end_point = (int(end.x * width), int(end.y * height))
            cv2.line(frame, start_point, end_point, (93, 184, 243), 2, cv2.LINE_AA)

        for landmark in hand_landmarks:
            point = (int(landmark.x * width), int(landmark.y * height))
            cv2.circle(frame, point, 4, (242, 210, 145), -1, cv2.LINE_AA)

    @staticmethod
    def _ensure_model() -> Path:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        if MODEL_PATH.exists():
            return MODEL_PATH

        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        return MODEL_PATH

    def _annotate_frame(self, frame: np.ndarray, gesture: int, label: str) -> None:
        overlay = frame.copy()
        cv2.rectangle(overlay, (14, 14), (420, 112), (18, 10, 7), -1)
        frame[:] = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

        status = self.snapshot()
        overlay_label = {
            0: "Show 1 or 2 fingers",
            1: "One finger detected",
            2: "Two fingers detected",
        }[gesture]
        overlay_scene = {
            "idle": "Scene: idle",
            "one": "Scene: first song",
            "two_wait": "Scene: correct kuy",
            "milk": "Scene: milk flow",
        }.get(status["scene"], "Scene: idle")
        lines = [
            "MediaPipe Gesture Camera",
            overlay_label,
            overlay_scene,
        ]
        if status["milk_countdown"] > 0:
            lines.append(f"Milk in: {status['milk_countdown']:.1f}s")

        for idx, text in enumerate(lines):
            cv2.putText(
                frame,
                text,
                (28, 42 + idx * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (242, 225, 184),
                2,
                cv2.LINE_AA,
            )

        color = (68, 168, 120) if gesture == 2 else (54, 131, 196) if gesture == 1 else (164, 138, 86)
        cv2.rectangle(frame, (18, 128), (210, 146), color, -1)
        progress = int(192 * status["hold_progress"])
        cv2.rectangle(frame, (18, 128), (18 + progress, 146), (232, 195, 99), -1)

    @staticmethod
    def _build_placeholder(title: str, subtitle: str) -> bytes:
        canvas = np.zeros((540, 960, 3), dtype=np.uint8)
        canvas[:] = (20, 13, 10)
        cv2.rectangle(canvas, (60, 60), (900, 480), (49, 33, 23), 2)
        cv2.putText(
            canvas,
            title,
            (94, 218),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (241, 214, 160),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            subtitle,
            (94, 268),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (194, 170, 126),
            2,
            cv2.LINE_AA,
        )
        encoded = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 84])[1]
        return encoded.tobytes()


app = Flask(__name__, static_folder=None)
engine = GestureEngine()


@app.get("/")
def index() -> Response:
    return send_file(BASE_DIR / "index.html")


@app.get("/api/state")
def api_state() -> Response:
    return jsonify(engine.snapshot())


@app.get("/video_feed")
def video_feed() -> Response:
    return Response(
        engine.mjpeg_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/media/<track_key>")
def media(track_key: str):
    meta = TRACKS.get(track_key)
    if not meta:
        return ("Unknown track", 404)

    path = meta["path"]
    if not path.exists():
        return ("Track not found", 404)

    return send_file(path, conditional=True, etag=True)


@app.get("/health")
def health() -> Response:
    return jsonify({"ok": True, "camera_ready": engine.snapshot()["camera_ready"]})


@atexit.register
def _shutdown_engine() -> None:
    engine.stop()


if __name__ == "__main__":
    engine.start()
    app.run(host="127.0.0.1", port=4173, debug=False, use_reloader=False, threaded=True)
