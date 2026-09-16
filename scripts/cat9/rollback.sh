#!/usr/bin/env bash
set -euo pipefail

NGINX_CONFIG="nginx/nginx.conf"
NGINX_CONTAINER="fraud-cat9-nginx"

echo "Rolling back canary deployment..."

python3 - "$NGINX_CONFIG" <<'PY'
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
text = config_path.read_text()

start_marker = "    # CAT9_TRAFFIC_SPLIT_START"
end_marker = "    # CAT9_TRAFFIC_SPLIT_END"

start = text.find(start_marker)
end = text.find(end_marker)

if start == -1 or end == -1:
    raise SystemExit("ERROR: CAT9 traffic split markers not found.")

end += len(end_marker)

block = """    # CAT9_TRAFFIC_SPLIT_START
    split_clients $request_id $deployment {
        100% stable;
        * canary;
    }
    # CAT9_TRAFFIC_SPLIT_END"""

text = text[:start] + block + text[end:]
config_path.write_text(text)
PY

echo "Testing Nginx configuration..."

docker exec "$NGINX_CONTAINER" nginx -t

echo "Reloading Nginx..."

docker exec "$NGINX_CONTAINER" nginx -s reload

echo
echo "ROLLBACK COMPLETE"
echo "Stable: 100%"
echo "Canary: 0%"
