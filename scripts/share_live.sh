#!/usr/bin/env bash
# Share the ScholarRAG app running on this laptop through a temporary public link (Cloudflare quick tunnel).
#
#   scripts/share_live.sh                     # random 6-character access code
#   ACCESS_CODE=mycode scripts/share_live.sh  # choose the code
#   NO_CODE=1 scripts/share_live.sh           # no access code
#
# Stop with Ctrl+C: the app and the tunnel shut down and the link stops working at once.
# Needs cloudflared (brew install cloudflared) and GEMINI_API_KEY in .env. No Cloudflare account is needed.
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-8000}"
if [ "${NO_CODE:-0}" = "1" ]; then
  CODE=""
else
  # "|| true": tr berhenti dengan SIGPIPE saat head selesai, dan pipefail menganggapnya error
  CODE="${ACCESS_CODE:-$(LC_ALL=C tr -dc 'a-z0-9' </dev/urandom | head -c 6 || true)}"
fi
LOG_DIR="$(mktemp -d)"
APP_PID="" TUNNEL_PID="" AWAKE_PID=""

cleanup() {
  trap - EXIT INT TERM
  for pid in $TUNNEL_PID $APP_PID $AWAKE_PID; do kill "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
  rm -rf "$LOG_DIR"
  echo
  echo "Stopped: the link no longer works."
}
trap cleanup EXIT INT TERM

command -v cloudflared >/dev/null || { echo "cloudflared is not installed: brew install cloudflared"; exit 1; }
[ -f .env ] || { echo ".env not found (it must contain GEMINI_API_KEY)"; exit 1; }
if curl -s -m 2 "http://127.0.0.1:$PORT/api/health" >/dev/null; then
  echo "Port $PORT is already in use: stop the other server first, or run with PORT=8001"; exit 1
fi

echo "Starting the app (about 20 s)…"
ACCESS_CODE="$CODE" .venv/bin/uvicorn app:app --app-dir src --host 127.0.0.1 --port "$PORT" >"$LOG_DIR/app.log" 2>&1 &
APP_PID=$!
for _ in $(seq 1 90); do
  curl -s -m 2 "http://127.0.0.1:$PORT/api/health" >/dev/null && break
  sleep 1
done
curl -s -m 2 "http://127.0.0.1:$PORT/api/health" >/dev/null || { echo "The app did not start:"; tail -20 "$LOG_DIR/app.log"; exit 1; }

echo "Opening the tunnel…"
cloudflared tunnel --no-autoupdate --url "http://127.0.0.1:$PORT" >"$LOG_DIR/tunnel.log" 2>&1 &
TUNNEL_PID=$!
URL=""
for _ in $(seq 1 60); do
  URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG_DIR/tunnel.log" | grep -v '//api\.' | head -1 || true)
  [ -n "$URL" ] && break
  sleep 1
done
[ -n "$URL" ] || { echo "The tunnel did not open:"; tail -20 "$LOG_DIR/tunnel.log"; exit 1; }

# jaga Mac tetap terjaga selama skrip ini berjalan
caffeinate -ims -w $$ &
AWAKE_PID=$!

echo
if [ -n "$CODE" ]; then
  echo "  Link to share:  $URL/?code=$CODE"
  echo "  Access code:    $CODE (already filled in by the link)"
else
  echo "  Link to share:  $URL/"
fi
echo "  The link can take up to a minute to start working. Press Ctrl+C to stop sharing."
echo
wait "$TUNNEL_PID"
