#!/usr/bin/env bash
set -euo pipefail

# Provider-specific bootstrap only. The model server contract remains the same
# OpenAI-compatible transport consumed by the local control plane.
EVIDENCE_ROOT=/opt/jlens-evidence
MODEL_REPOSITORY=cyankiwi/Qwen3.5-4B-AWQ-4bit
MODEL_REVISION=ef85d23bebaba87b3c4672ba11c449c79dbdb23e
BASE_MODEL_REPOSITORY=Qwen/Qwen3.5-4B
VLLM_IMAGE='vllm/vllm-openai@sha256:9238dec7203672c9cb9f9f6989d7b2147a6e34eade8743ba388fd688cfe61f51'
MODEL_ROOT=/opt/jlens-model/Qwen3.5-4B-AWQ-ef85d23
VLLM_CACHE=/opt/jlens-vllm-cache
INSTANCE_ID=${INSTANCE_ID:?INSTANCE_ID is required}
REGION=${REGION:?REGION is required}
ZONE=${ZONE:?ZONE is required}

install -d -m 0755 "$EVIDENCE_ROOT/runtime" "$EVIDENCE_ROOT/server-logs" \
  "$VLLM_CACHE"
exec > >(tee -a "$EVIDENCE_ROOT/server-logs/bootstrap.log") 2>&1

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y ca-certificates curl docker.io gnupg jq

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
  >/etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl enable --now docker
systemctl restart docker

docker pull "$VLLM_IMAGE"
docker run --rm --gpus all --entrypoint nvidia-smi "$VLLM_IMAGE" \
  --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader \
  | tee "$EVIDENCE_ROOT/runtime/container-gpu-smoke.csv"
install -d -m 0755 "$MODEL_ROOT"
docker run --rm --entrypoint /usr/bin/python3 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v /opt/jlens-model:/opt/jlens-model \
  "$VLLM_IMAGE" -c \
  "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${MODEL_REPOSITORY}', revision='${MODEL_REVISION}', local_dir='${MODEL_ROOT}')"

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
ExecStartPre=/usr/bin/docker run --rm --gpus all --entrypoint nvidia-smi ${VLLM_IMAGE} --query-gpu=name,uuid,driver_version --format=csv,noheader
ExecStart=/usr/bin/docker run --name jlens-model-worker --gpus all --ipc=host \\
  -p 127.0.0.1:8000:8000 \\
  -v ${MODEL_ROOT}:${MODEL_ROOT}:ro \\
  -v ${EVIDENCE_ROOT}:/evidence \\
  -v ${VLLM_CACHE}:/root/.cache \\
  ${VLLM_IMAGE} \\
  --model ${MODEL_ROOT} \\
  --tokenizer ${MODEL_ROOT} \\
  --served-model-name same-base-cuda \\
  --host 0.0.0.0 --port 8000 \\
  --language-model-only \\
  --dtype float16 \\
  --max-model-len 16384 \\
  --gpu-memory-utilization 0.90 \\
  --enforce-eager \\
  --gdn-prefill-backend triton \\
  --generation-config vllm
ExecStop=/usr/bin/docker stop jlens-model-worker
StandardOutput=append:${EVIDENCE_ROOT}/server-logs/vllm.stdout.log
StandardError=append:${EVIDENCE_ROOT}/server-logs/vllm.stderr.log

[Install]
WantedBy=multi-user.target
UNIT

cat >"$EVIDENCE_ROOT/runtime/BOOTSTRAP-IDENTITY.txt" <<IDENTITY
provider=tencent-cloud
region=${REGION}
zone=${ZONE}
instance_id=${INSTANCE_ID}
model_repository=${MODEL_REPOSITORY}
model_revision=${MODEL_REVISION}
base_model_repository=${BASE_MODEL_REPOSITORY}
model_root=${MODEL_ROOT}
container_image=${VLLM_IMAGE}
endpoint_bind=127.0.0.1:8000
quantization=compressed-tensors-int4-group32
dtype=float16-activations
max_model_len=16384
gpu_memory_utilization=0.90
execution_mode=eager
gdn_prefill_backend=triton
cache_root=${VLLM_CACHE}
IDENTITY
sha256sum /etc/systemd/system/jlens-model-worker.service \
  "$EVIDENCE_ROOT/runtime/BOOTSTRAP-IDENTITY.txt" \
  "$EVIDENCE_ROOT/runtime/container-gpu-smoke.csv" \
  >"$EVIDENCE_ROOT/runtime/BOOTSTRAP-SHA256SUMS.txt"

cat >/usr/local/sbin/jlens-freeze-runtime-identity <<'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail
EVIDENCE_ROOT=/opt/jlens-evidence
MODEL_ROOT=/opt/jlens-model/Qwen3.5-4B-AWQ-ef85d23
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
TimeoutStartSec=0
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --now jlens-model-worker.service
systemctl enable jlens-runtime-identity.service
systemctl start --no-block jlens-runtime-identity.service
