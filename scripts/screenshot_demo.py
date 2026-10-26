"""Take README screenshots of the recorded demo with headless Chrome (light theme, answer area only).

Serve the static demo first, then run this script from the project root:
    .venv/bin/python scripts/build_space.py --static
    (cd dist/static-space && ../../.venv/bin/python -m http.server 8010 --bind 127.0.0.1)
    .venv/bin/python scripts/screenshot_demo.py

Chrome runs with a throwaway profile, so the browser profile on this machine is not touched.
"""
import base64
import json
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import websocket

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
URL = "http://127.0.0.1:8010/index.html"
PORT = 9333
OUT = Path(__file__).resolve().parent.parent / "docs"
# (nomor rekaman, file, tinggi maksimum dalam piksel CSS)
SHOTS = [(1, "demo-answer.png", 760), (6, "demo-refusal.png", 900)]


class CDP:
    def __init__(self, ws_url):
        # tanpa header Origin, Chrome menerima koneksi tanpa perlu --remote-allow-origins
        self.ws, self.n = websocket.create_connection(ws_url, timeout=30, suppress_origin=True), 0

    def call(self, method, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.n:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg.get("result", {})


def main():
    OUT.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as profile:
        chrome = subprocess.Popen([CHROME, "--headless=new", "--disable-gpu", "--no-first-run", "--hide-scrollbars",
                                   f"--remote-debugging-port={PORT}", f"--user-data-dir={profile}", "about:blank"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            for _ in range(50):
                try:
                    targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/list"))
                    page = next(t for t in targets if t["type"] == "page")
                    break
                except Exception:
                    time.sleep(0.2)
            cdp = CDP(page["webSocketDebuggerUrl"])
            cdp.call("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": "light"}])
            cdp.call("Emulation.setDeviceMetricsOverride", width=900, height=1400, deviceScaleFactor=2, mobile=False)
            for idx, name, max_h in SHOTS:
                cdp.call("Page.navigate", url="about:blank")
                time.sleep(0.3)
                cdp.call("Page.navigate", url=f"{URL}#q={idx}")
                time.sleep(2.0)   # tunggu halaman selesai digambar
                rect = json.loads(cdp.call("Runtime.evaluate", returnByValue=True, expression=
                    "window.scrollTo(0, 0); const r = document.getElementById('out').getBoundingClientRect();"
                    "JSON.stringify({x: r.x, y: r.y + window.scrollY, w: r.width, h: r.height})")["result"]["value"])
                pad = 16
                clip = {"x": max(0, rect["x"] - pad), "y": max(0, rect["y"] - pad), "width": rect["w"] + 2 * pad,
                        "height": min(rect["h"], max_h) + 2 * pad, "scale": 1}
                img = cdp.call("Page.captureScreenshot", format="png", clip=clip, captureBeyondViewport=True)["data"]
                (OUT / name).write_bytes(base64.b64decode(img))
                print(f"docs/{name}: {clip['width']:.0f}x{clip['height']:.0f} px (CSS)")
        finally:
            chrome.terminate()
            chrome.wait(timeout=10)


if __name__ == "__main__":
    main()
