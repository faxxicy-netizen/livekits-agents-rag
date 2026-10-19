#!/usr/bin/env bash
set -euo pipefail
: "${PUBLIC_HOST:?Set PUBLIC_HOST to the public IP or domain used by your certificate}"
if [[ ! "$PUBLIC_HOST" =~ ^[A-Za-z0-9][A-Za-z0-9.-]*$ ]]; then
  echo 'PUBLIC_HOST must be an IPv4 address or domain name.' >&2
  exit 1
fi
for name in livekit qdrant; do
  install -d "/opt/support-tools/${name}"
  tar -xzf "/opt/support-tools/${name}.tar.gz" -C "/opt/support-tools/${name}"
done
chown -R support:support /opt/support-agent /var/lib/support-agent
chmod 700 /var/lib/support-agent
if [ -f /var/lib/support-agent/support.db ]; then chmod 600 /var/lib/support-agent/support.db; fi
chmod 600 /opt/support-agent/.env /opt/support-agent/deploy/livekit.yaml
for service in qdrant livekit api agent; do
  case "$service" in
    qdrant) command='/opt/support-tools/qdrant/qdrant --config-path /opt/support-agent/deploy/qdrant.yaml'; after='network-online.target';;
    livekit) command='/opt/support-tools/livekit/livekit-server --config /opt/support-agent/deploy/livekit.yaml'; after='network-online.target';;
    api) command='/opt/support-agent/.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --proxy-headers --forwarded-allow-ips 127.0.0.1'; after='support-qdrant.service support-livekit.service';;
    agent) command='/opt/support-agent/.venv/bin/python -m app.voice start'; after='support-api.service support-livekit.service';;
  esac
  cat >"/etc/systemd/system/support-${service}.service" <<EOF
[Unit]
Description=Zhisheng demo ${service}
After=network-online.target ${after}
Wants=network-online.target
[Service]
User=support
Group=support
WorkingDirectory=/opt/support-agent
Environment=HOME=/var/lib/support-agent
Environment=HF_HOME=/var/lib/support-agent/cache
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONUTF8=1
ExecStart=${command}
Restart=on-failure
RestartSec=3
TimeoutStopSec=20
LimitNOFILE=65536
NoNewPrivileges=true
PrivateTmp=true
UMask=0077
[Install]
WantedBy=multi-user.target
EOF
done
for service in api agent; do
  install -d "/etc/systemd/system/support-${service}.service.d"
  if [ "$service" = api ]; then endpoint='http://127.0.0.1:6333/readyz';
  else endpoint='http://127.0.0.1:8000/healthz'; fi
  cat >"/etc/systemd/system/support-${service}.service.d/readiness.conf" <<EOF
[Service]
ExecStartPre=/usr/bin/curl --fail --silent --retry 30 --retry-connrefused --retry-delay 1 --max-time 2 ${endpoint}
EOF
done
cat >/etc/nginx/conf.d/support.conf <<'CONF'
map $http_upgrade $connection_upgrade { default upgrade; '' close; }
server {
 listen 80 default_server;
 server_name __PUBLIC_HOST__;
 location ^~ /.well-known/acme-challenge/ { root /var/www/acme; }
 location / { return 301 https://__PUBLIC_HOST__$request_uri; }
}
server {
 listen 443 ssl;
 server_name __PUBLIC_HOST__;
 ssl_certificate /etc/letsencrypt/live/__PUBLIC_HOST__/fullchain.pem;
 ssl_certificate_key /etc/letsencrypt/live/__PUBLIC_HOST__/privkey.pem;
 ssl_protocols TLSv1.2 TLSv1.3;
 client_max_body_size 12m;
 location /rtc {
   proxy_pass http://127.0.0.1:7880;
   proxy_http_version 1.1;
   proxy_set_header Upgrade $http_upgrade;
   proxy_set_header Connection $connection_upgrade;
   proxy_set_header Host $host;
   proxy_set_header X-Forwarded-For $remote_addr;
   proxy_set_header X-Forwarded-Proto https;
   proxy_read_timeout 3600s;
   proxy_send_timeout 3600s;
   proxy_buffering off;
 }
 location / {
   proxy_pass http://127.0.0.1:8000;
   proxy_http_version 1.1;
   proxy_set_header Host $host;
   proxy_set_header X-Forwarded-For $remote_addr;
   proxy_set_header X-Forwarded-Proto https;
   proxy_read_timeout 120s;
 }
}
CONF
sed -i "s/__PUBLIC_HOST__/${PUBLIC_HOST}/g" /etc/nginx/conf.d/support.conf
if command -v getenforce >/dev/null && [ "$(getenforce)" = Enforcing ]; then
  setsebool -P httpd_can_network_connect 1
fi
cat >/etc/systemd/system/support-cert-renew.service <<'EOF'
[Unit]
Description=Renew short-lived IP TLS certificate
[Service]
Type=oneshot
ExecStart=/opt/support-tools/python/bin/certbot renew --quiet --deploy-hook "systemctl reload nginx"
EOF
cat >/etc/systemd/system/support-cert-renew.timer <<'EOF'
[Unit]
Description=Check IP certificate renewal twice a day
[Timer]
OnCalendar=*-*-* 00,12:15:00
RandomizedDelaySec=1800
Persistent=true
[Install]
WantedBy=timers.target
EOF
nginx -t
systemctl daemon-reload
systemctl enable --now support-qdrant support-livekit support-api support-agent support-cert-renew.timer
systemctl reload nginx
echo 'Services installed'
