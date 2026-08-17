#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${1:?usage: checkpoint-stop.sh REPO_ROOT}
INSTANCE_ID=${INSTANCE_ID:?set INSTANCE_ID to the target Tencent CVM instance}
REGION=${REGION:-ap-hongkong}
RUN_ROOT=${RUN_ROOT:-"$REPO_ROOT/artifacts/v2.5.0/v2.5.0-local-jlens/runs/skill-evolution-loop/aws-migration-r075"}
COMMAND_NAME=${COMMAND_NAME:-jlens-checkpoint-runtime}
FINAL_ACTION=${FINAL_ACTION:-stop}
CHECKPOINT_ROOT="$RUN_ROOT/tencent/checkpoints"
TIMESTAMP=$(date -u '+%Y%m%dT%H%M%SZ')
DESTINATION="$CHECKPOINT_ROOT/$TIMESTAMP"

if [[ "$FINAL_ACTION" != stop ]]; then
  printf 'FINAL_ACTION must be stop; instance termination is prohibited\n' >&2
  exit 2
fi

test -n "${TENCENTCLOUD_SECRET_ID:-}"
test -n "${TENCENTCLOUD_SECRET_KEY:-}"
install -d -m 0755 "$DESTINATION"

remote='set -eu
curl -fsS http://127.0.0.1:8000/health >/dev/null
test -f /opt/jlens-evidence/runtime/RUNTIME-IDENTITY-SHA256SUMS.txt
tar -C /opt/jlens-evidence -czf - runtime'
content=$(printf '%s' "$remote" | base64 | tr -d '\n')
invocation=$(tccli tat RunCommand \
  --region "$REGION" \
  --Content "$content" \
  --InstanceIds "[\"$INSTANCE_ID\"]" \
  --CommandName "$COMMAND_NAME" \
  --CommandType SHELL \
  --Timeout 300)
invocation_id=$(printf '%s' "$invocation" | jq -er .InvocationId)

for _ in $(seq 1 90); do
  task=$(tccli tat DescribeInvocationTasks \
    --region "$REGION" \
    --Filters "[{\"Name\":\"invocation-id\",\"Values\":[\"$invocation_id\"]}]" \
    --HideOutput False)
  status=$(printf '%s' "$task" | jq -er '.InvocationTaskSet[0].TaskStatus')
  case "$status" in
    SUCCESS)
      printf '%s' "$task" | jq -er '.InvocationTaskSet[0].TaskResult.Output' \
        | base64 -d >"$DESTINATION/runtime-evidence.tar.gz"
      break
      ;;
    FAILED|TIMEOUT|CANCELLED|TERMINATED)
      printf '%s\n' "remote checkpoint failed: $status" >&2
      exit 1
      ;;
  esac
  sleep 2
done

test -s "$DESTINATION/runtime-evidence.tar.gz"
tar -tzf "$DESTINATION/runtime-evidence.tar.gz" >"$DESTINATION/CONTENTS.txt"
tar -xOf "$DESTINATION/runtime-evidence.tar.gz" \
  runtime/RUNTIME-IDENTITY-SHA256SUMS.txt >/dev/null
shasum -a 256 "$DESTINATION/runtime-evidence.tar.gz" \
  "$DESTINATION/CONTENTS.txt" >"$DESTINATION/SHA256SUMS.txt"

temporary="$DESTINATION/CHECKPOINT.json.tmp"
jq -n \
  --arg recorded_at "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
  --arg instance_id "$INSTANCE_ID" \
  --arg invocation_id "$invocation_id" \
  --arg archive_sha256 "$(shasum -a 256 "$DESTINATION/runtime-evidence.tar.gz" | awk '{print $1}')" \
  '{schema_version:1,recorded_at:$recorded_at,instance_id:$instance_id,invocation_id:$invocation_id,checkpoint_atomic:true,append_only_evidence_synced:true,recovery_verified:true,runtime_archive_sha256:$archive_sha256}' \
  >"$temporary"
mv "$temporary" "$DESTINATION/CHECKPOINT.json"

tccli cvm StopInstances \
  --region "$REGION" \
  --InstanceIds "[\"$INSTANCE_ID\"]" \
  --StopType SOFT_FIRST \
  --StoppedMode STOP_CHARGING \
  >"$DESTINATION/STOP-REQUEST.json"

for _ in $(seq 1 90); do
  state=$(tccli cvm DescribeInstances \
    --region "$REGION" \
    --InstanceIds "[\"$INSTANCE_ID\"]")
  printf '%s\n' "$state" >"$DESTINATION/INSTANCE-STATE.json.tmp"
  current=$(printf '%s' "$state" | jq -er '.InstanceSet[0].InstanceState')
  if [[ "$current" == STOPPED ]]; then
    mv "$DESTINATION/INSTANCE-STATE.json.tmp" "$DESTINATION/INSTANCE-STATE.json"
    break
  fi
  sleep 2
done
test -f "$DESTINATION/INSTANCE-STATE.json"
printf '%s\n' "$DESTINATION"
