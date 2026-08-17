#!/usr/bin/env bash
set -euo pipefail

EVIDENCE_ROOT=/opt/jlens-evidence
MODEL_REPOSITORY=Qwen/Qwen3.5-4B
MODEL_REVISION=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
VLLM_IMAGE='vllm/vllm-openai@sha256:9238dec7203672c9cb9f9f6989d7b2147a6e34eade8743ba388fd688cfe61f51'

install -d -m 0755 "$EVIDENCE_ROOT/runtime" "$EVIDENCE_ROOT/server-logs"
exec > >(tee -a "$EVIDENCE_ROOT/server-logs/cloud-init.log") 2>&1

systemctl enable --now docker
docker pull "$VLLM_IMAGE"

cat >/etc/systemd/system/jlens-model-worker.service <<UNIT
[Unit]
Description=JLENS generic OpenAI-compatible CUDA model worker
After=docker.service network-online.target
Wants=network-online.target

[Service]
Type=simple
Restart=on-failure
RestartSec=15
ExecStartPre=-/usr/bin/docker rm -f jlens-model-worker
ExecStart=/usr/bin/docker run --name jlens-model-worker --gpus all --ipc=host \\
  -p 127.0.0.1:8000:8000 \\
  -v /root/.cache/huggingface:/root/.cache/huggingface \\
  -v ${EVIDENCE_ROOT}:/evidence \\
  ${VLLM_IMAGE} \\
  --model ${MODEL_REPOSITORY} \\
  --revision ${MODEL_REVISION} \\
  --served-model-name same-base-cuda \\
  --host 0.0.0.0 --port 8000 \\
  --language-model-only \\
  --dtype float16 \\
  --max-model-len 32768 \\
  --gpu-memory-utilization 0.90 \\
  --generation-config vllm \\
  --download-dir /root/.cache/huggingface
ExecStop=/usr/bin/docker stop jlens-model-worker
StandardOutput=append:${EVIDENCE_ROOT}/server-logs/vllm.stdout.log
StandardError=append:${EVIDENCE_ROOT}/server-logs/vllm.stderr.log

[Install]
WantedBy=multi-user.target
UNIT

IMDS_TOKEN=$(curl -fsS -X PUT \
  -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' \
  http://169.254.169.254/latest/api/token)
imds() {
  curl -fsS -H "X-aws-ec2-metadata-token: ${IMDS_TOKEN}" \
    "http://169.254.169.254/latest/meta-data/$1"
}
cat >"$EVIDENCE_ROOT/runtime/BOOTSTRAP-IDENTITY.txt" <<IDENTITY
model_repository=${MODEL_REPOSITORY}
model_revision=${MODEL_REVISION}
container_image=${VLLM_IMAGE}
instance_id=$(imds instance-id)
ami_id=$(imds ami-id)
instance_type=$(imds instance-type)
IDENTITY
sha256sum /etc/systemd/system/jlens-model-worker.service \
  "$EVIDENCE_ROOT/runtime/BOOTSTRAP-IDENTITY.txt" \
  >>"$EVIDENCE_ROOT/runtime/ARTIFACT-SHA256SUMS.txt"

systemctl daemon-reload
systemctl enable --now jlens-model-worker.service

cat >/usr/local/sbin/jlens-freeze-runtime-identity <<'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail
EVIDENCE_ROOT=/opt/jlens-evidence
MODEL_ROOT=$(find /root/.cache/huggingface/hub \
  -path '*/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a' -type d -print -quit)
test -n "$MODEL_ROOT"
test -f "$MODEL_ROOT/config.json"
test -f "$MODEL_ROOT/tokenizer_config.json"
test -f "$MODEL_ROOT/chat_template.jinja"
install -d -m 0755 "$EVIDENCE_ROOT/runtime/model-files"
find -L "$MODEL_ROOT" -maxdepth 1 -type f \
  \( -name '*.safetensors' -o -name 'config.json' -o -name 'tokenizer.json' \
     -o -name 'tokenizer_config.json' -o -name 'chat_template.jinja' \) \
  -print0 | sort -z | xargs -0 sha256sum \
  >"$EVIDENCE_ROOT/runtime/model-files/SHA256SUMS.txt"
docker inspect --format '{{.Image}}' jlens-model-worker \
  >"$EVIDENCE_ROOT/runtime/container-image-id.txt"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total \
  --format=csv,noheader \
  >"$EVIDENCE_ROOT/runtime/gpu-identity.csv"
curl -fsS http://127.0.0.1:8000/v1/models \
  >"$EVIDENCE_ROOT/runtime/openai-models.json"
sha256sum \
  "$EVIDENCE_ROOT/runtime/model-files/SHA256SUMS.txt" \
  "$EVIDENCE_ROOT/runtime/container-image-id.txt" \
  "$EVIDENCE_ROOT/runtime/gpu-identity.csv" \
  "$EVIDENCE_ROOT/runtime/openai-models.json" \
  >"$EVIDENCE_ROOT/runtime/RUNTIME-IDENTITY-SHA256SUMS.txt"
chmod -R a-w "$EVIDENCE_ROOT/runtime/model-files" \
  "$EVIDENCE_ROOT/runtime/container-image-id.txt" \
  "$EVIDENCE_ROOT/runtime/gpu-identity.csv" \
  "$EVIDENCE_ROOT/runtime/openai-models.json" \
  "$EVIDENCE_ROOT/runtime/RUNTIME-IDENTITY-SHA256SUMS.txt"
SCRIPT
chmod 0755 /usr/local/sbin/jlens-freeze-runtime-identity

cat >/etc/systemd/system/jlens-runtime-identity.service <<UNIT
[Unit]
Description=Freeze JLENS CUDA runtime identity after model readiness
After=jlens-model-worker.service
Requires=jlens-model-worker.service

[Service]
Type=oneshot
ExecStart=/bin/bash -c 'for i in {1..240}; do curl -fsS http://127.0.0.1:8000/health && exec /usr/local/sbin/jlens-freeze-runtime-identity; sleep 15; done; exit 1'
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now jlens-runtime-identity.service
