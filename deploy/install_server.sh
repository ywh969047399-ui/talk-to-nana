#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/opt/talk-to-fengge"
CONFIG_DIR="/etc/talk-to-fengge"
API_DOMAIN="api.45-32-18-32.sslip.io"
LIVEKIT_DOMAIN="livekit.45-32-18-32.sslip.io"
PUBLIC_IP="45.32.18.32"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this script as root." >&2
  exit 1
fi

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  ca-certificates certbot curl nginx openssl python3-certbot-nginx

if ! command -v livekit-server >/dev/null 2>&1; then
  curl -sSL https://get.livekit.io | bash
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
fi

if ! id talkfengge >/dev/null 2>&1; then
  useradd --system --home-dir "${APP_DIR}" --shell /usr/sbin/nologin talkfengge
fi

install -d -o talkfengge -g talkfengge "${APP_DIR}" "${CONFIG_DIR}"
chown -R talkfengge:talkfengge "${APP_DIR}"

cd "${APP_DIR}"
sudo -u talkfengge uv sync --python 3.13 --no-dev

if [[ ! -f "${CONFIG_DIR}/livekit.yaml" ]]; then
  LIVEKIT_API_KEY="lk_$(openssl rand -hex 8)"
  LIVEKIT_API_SECRET="$(openssl rand -base64 48 | tr -d '\n/+=' | cut -c1-48)"
  cat > "${CONFIG_DIR}/livekit.yaml" <<EOF
port: 7880
bind_addresses:
  - "0.0.0.0"
rtc:
  tcp_port: 7881
  udp_port: 7882
  use_external_ip: false
  node_ip: "${PUBLIC_IP}"
keys:
  ${LIVEKIT_API_KEY}: "${LIVEKIT_API_SECRET}"
logging:
  level: info
  json: true
EOF
else
  LIVEKIT_API_KEY="$(sed -nE 's/^  ([^:]+):.*/\1/p' "${CONFIG_DIR}/livekit.yaml" | head -1)"
  LIVEKIT_API_SECRET="$(sed -nE 's/^  [^:]+: "([^"]+)"/\1/p' "${CONFIG_DIR}/livekit.yaml" | head -1)"
fi

python3 - "${APP_DIR}/.env.local" "${LIVEKIT_API_KEY}" "${LIVEKIT_API_SECRET}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
updates = {
    "LIVEKIT_URL": "wss://livekit.45-32-18-32.sslip.io",
    "LIVEKIT_API_KEY": sys.argv[2],
    "LIVEKIT_API_SECRET": sys.argv[3],
    "EGRESS_PROXY_URL": "",
    "WEB_PORT": "8766",
    "LIVEKIT_WORKER_PORT": "8081",
}
lines = path.read_text(encoding="utf-8").splitlines()
seen = set()
result = []
for line in lines:
    key = line.split("=", 1)[0] if "=" in line else ""
    if key in updates:
        result.append(f"{key}={updates[key]}")
        seen.add(key)
    else:
        result.append(line)
for key, value in updates.items():
    if key not in seen:
        result.append(f"{key}={value}")
path.write_text("\n".join(result) + "\n", encoding="utf-8")
PY

chown talkfengge:talkfengge "${APP_DIR}/.env.local"
chmod 600 "${APP_DIR}/.env.local" "${CONFIG_DIR}/livekit.yaml"

cat > /etc/systemd/system/talk-livekit.service <<EOF
[Unit]
Description=Talk to Fengge LiveKit server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=talkfengge
Group=talkfengge
ExecStart=/usr/local/bin/livekit-server --config ${CONFIG_DIR}/livekit.yaml
Restart=always
RestartSec=3
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/talk-web.service <<EOF
[Unit]
Description=Talk to Fengge token API
After=network-online.target talk-livekit.service
Wants=network-online.target

[Service]
Type=simple
User=talkfengge
Group=talkfengge
WorkingDirectory=${APP_DIR}
Environment=PYTHONUTF8=1
Environment=PYTHONUNBUFFERED=1
ExecStart=${APP_DIR}/.venv/bin/python -X utf8 -m worker.web_server
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/talk-worker.service <<EOF
[Unit]
Description=Talk to Fengge voice worker
After=network-online.target talk-livekit.service
Wants=network-online.target

[Service]
Type=simple
User=talkfengge
Group=talkfengge
WorkingDirectory=${APP_DIR}
Environment=PYTHONUTF8=1
Environment=PYTHONUNBUFFERED=1
ExecStart=${APP_DIR}/.venv/bin/python -X utf8 -m worker.main start
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

cat > /etc/nginx/sites-available/talk-to-fengge <<EOF
server {
    listen 80;
    listen [::]:80;
    server_name ${API_DOMAIN} ${LIVEKIT_DOMAIN};
    location / { return 404; }
}
EOF
ln -sfn /etc/nginx/sites-available/talk-to-fengge /etc/nginx/sites-enabled/talk-to-fengge
nginx -t
systemctl reload nginx

if [[ ! -d /etc/letsencrypt/live/talk-to-fengge ]]; then
  certbot certonly --nginx --non-interactive --agree-tos \
    --register-unsafely-without-email --cert-name talk-to-fengge \
    -d "${API_DOMAIN}" -d "${LIVEKIT_DOMAIN}"
fi

cat > /etc/nginx/sites-available/talk-to-fengge <<EOF
server {
    listen 80;
    listen [::]:80;
    server_name ${API_DOMAIN} ${LIVEKIT_DOMAIN};
    return 301 https://\$host\$request_uri;
}

server {
    listen 443 ssl;
    listen [::]:443 ssl;
    server_name ${API_DOMAIN};

    ssl_certificate /etc/letsencrypt/live/talk-to-fengge/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/talk-to-fengge/privkey.pem;
    include /etc/letsencrypt/options-ssl-nginx.conf;
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;

    location / {
        proxy_pass http://127.0.0.1:8766;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}

server {
    listen 443 ssl;
    listen [::]:443 ssl;
    server_name ${LIVEKIT_DOMAIN};

    ssl_certificate /etc/letsencrypt/live/talk-to-fengge/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/talk-to-fengge/privkey.pem;
    include /etc/letsencrypt/options-ssl-nginx.conf;
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;

    location / {
        proxy_pass http://127.0.0.1:7880;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_read_timeout 86400;
        proxy_send_timeout 86400;
    }
}
EOF

nginx -t
systemctl reload nginx
ufw allow 7881/tcp
ufw allow 7882/udp
systemctl daemon-reload
systemctl enable --now talk-livekit talk-web talk-worker

sleep 5
curl -fsS http://127.0.0.1:8766/health
echo
systemctl --no-pager --full status talk-livekit talk-web talk-worker | sed -n '1,90p'
