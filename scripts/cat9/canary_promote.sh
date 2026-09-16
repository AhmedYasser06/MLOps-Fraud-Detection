#!/usr/bin/env bash
set -euo pipefail

NGINX_CONFIG="nginx/nginx.conf"
NGINX_CONTAINER="fraud-cat9-nginx"

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 {10|25|50|100}"
    exit 1
fi

CANARY_PERCENT="$1"

case "$CANARY_PERCENT" in
    10|25|50|100)
        ;;
    *)
        echo "ERROR: Canary percentage must be 10, 25, 50, or 100."
        exit 1
        ;;
esac

if [[ "$CANARY_PERCENT" == "100" ]]; then
    STABLE_PERCENT=0
else
    STABLE_PERCENT=$((100 - CANARY_PERCENT))
fi

echo "Promoting canary to ${CANARY_PERCENT}%..."
echo "Stable traffic: ${STABLE_PERCENT}%"
echo "Canary traffic: ${CANARY_PERCENT}%"

python3 - "$NGINX_CONFIG" "$STABLE_PERCENT" <<'PY'
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
stable_percent = int(sys.argv[2])

text = config_path.read_text()

start_marker = "    # CAT9_TRAFFIC_SPLIT_START"
end_marker = "    # CAT9_TRAFFIC_SPLIT_END"

start = text.find(start_marker)
end = text.find(end_marker)

if start == -1 or end == -1:
    raise SystemExit("ERROR: CAT9 traffic split markers not found.")

end += len(end_marker)

if stable_percent == 0:
    block = """    # CAT9_TRAFFIC_SPLIT_START
    split_clients $request_id $deployment {
        * canary;
    }
    # CAT9_TRAFFIC_SPLIT_END"""
else:
    block = f"""    # CAT9_TRAFFIC_SPLIT_START
    split_clients $request_id $deployment {{
        {stable_percent}% stable;
        * canary;
    }}
    # CAT9_TRAFFIC_SPLIT_END"""

text = text[:start] + block + text[end:]
config_path.write_text(text)
PY

echo "Testing Nginx configuration..."

docker exec "$NGINX_CONTAINER" nginx -t

echo "Reloading Nginx..."

docker exec "$NGINX_CONTAINER" nginx -s reload

echo
echo "CAT9 canary is now ${CANARY_PERCENT}%."
