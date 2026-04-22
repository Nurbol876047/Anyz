from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, Response, jsonify, send_file

BASE_DIR = Path(__file__).resolve().parent

TRACKS = {
    "one": {
        "title": "1 саусақ: басқа әуен",
        "path": BASE_DIR / "track1.mp3",
    },
    "two": {
        "title": "2 саусақ: Төлеген Момбеков - Бозінген (күй)",
        "path": BASE_DIR / "Төлеген Момбеков - Бозінген (күй).mp3",
    },
}
VIDEO_CLIP = {
    "title": "Домбыра интерактивіне арналған видео",
    "path": BASE_DIR / "media" / "dombra_story.mp4",
}


def build_tracks_payload() -> dict[str, dict[str, str | bool]]:
    return {
        key: {
            "title": meta["title"],
            "available": meta["path"].exists(),
        }
        for key, meta in TRACKS.items()
    }


def build_client_state() -> dict:
    return {
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
        "status": (
            "Камераға рұқсат беріңіз. Қол қимылын тану енді браузердегі "
            "JavaScript MediaPipe арқылы жүреді."
        ),
        "tracks": build_tracks_payload(),
    }


app = Flask(__name__, static_folder=None)


@app.get("/")
def index() -> Response:
    return send_file(BASE_DIR / "index.html")


@app.get("/dombra")
def dombra_page() -> Response:
    return send_file(BASE_DIR / "dombra.html")


@app.get("/test")
def test_page() -> Response:
    return send_file(BASE_DIR / "test.html")


@app.get("/app.js")
def app_script() -> Response:
    return send_file(BASE_DIR / "app.js", mimetype="text/javascript")


@app.get("/dombra.js")
def dombra_script() -> Response:
    return send_file(BASE_DIR / "dombra.js", mimetype="text/javascript")


@app.get("/test.js")
def test_script() -> Response:
    return send_file(BASE_DIR / "test.js", mimetype="text/javascript")


@app.get("/api/state")
def api_state() -> Response:
    return jsonify(build_client_state())


@app.get("/media/<track_key>")
def media(track_key: str):
    meta = TRACKS.get(track_key)
    if not meta:
        return ("Unknown track", 404)

    path = meta["path"]
    if not path.exists():
        return ("Track not found", 404)

    return send_file(path, conditional=True, etag=True)


@app.get("/media/dombra-video")
def dombra_video():
    path = VIDEO_CLIP["path"]
    if not path.exists():
        return ("Video not found", 404)

    return send_file(path, conditional=True, etag=True)


@app.get("/health")
def health() -> Response:
    return jsonify(
        {
            "ok": True,
            "gesture_runtime": "browser-js-mediapipe",
            "tracks": build_tracks_payload(),
            "dombra_video": VIDEO_CLIP["path"].exists(),
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False, threaded=True)
