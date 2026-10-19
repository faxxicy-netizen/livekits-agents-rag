#!/usr/bin/env bash
set -euo pipefail
dnf -y install nginx python3-pip tar gzip curl >/var/log/support-bootstrap.log 2>&1
id support >/dev/null 2>&1 || useradd --system --home-dir /opt/support-agent --shell /sbin/nologin support
install -d -m 755 /opt/support-agent /opt/support-tools /var/lib/support-agent /var/www/acme
if [ ! -f /var/lib/support-agent.swap ] && [ "$(awk 'END{print NR}' /proc/swaps)" = 1 ]; then
  fallocate -l 1G /var/lib/support-agent.swap
  chmod 600 /var/lib/support-agent.swap
  mkswap /var/lib/support-agent.swap >/dev/null
  swapon /var/lib/support-agent.swap
  echo '/var/lib/support-agent.swap none swap sw 0 0' >> /etc/fstab
fi
python3 -m venv /opt/support-tools/python
/opt/support-tools/python/bin/pip install -q uv 'certbot>=5.4' >>/var/log/support-bootstrap.log 2>&1
for spec in 'livekit 1.13.10' 'qdrant 1.19.2'; do
  read -r name version <<<"$spec"
  if [ "$name" = livekit ]; then
    url="https://github.com/livekit/livekit/releases/download/v${version}/livekit_${version}_linux_amd64.tar.gz"
  else
    url="https://github.com/qdrant/qdrant/releases/download/v${version}/qdrant-x86_64-unknown-linux-musl.tar.gz"
  fi
  curl -fsSL --retry 2 --connect-timeout 15 --max-time 300 "$url" -o "/opt/support-tools/${name}.tar.gz"
  mkdir -p "/opt/support-tools/${name}"
  tar -xzf "/opt/support-tools/${name}.tar.gz" -C "/opt/support-tools/${name}"
done
echo 'Bootstrap complete'
