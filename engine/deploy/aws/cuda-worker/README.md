# Generic CUDA execution worker

This deployment-only bundle creates an OpenAI-compatible inference endpoint. The
core loop knows only `ModelTransport`; provider, instance, model serving, and SSH
tunnel details stay here.

The service binds to instance localhost. The local control plane connects through
an SSH tunnel, for example `127.0.0.1:18000 -> worker:127.0.0.1:8000`. Runtime
evidence is append-only under `/opt/jlens-evidence`; the instance root EBS volume
is retained on stop and explicitly deleted only during a reviewed rollback.

Pinned deployment identity:

- model: `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- server: pinned `nightly-x86_64` digest `vllm/vllm-openai@sha256:9238dec7203672c9cb9f9f6989d7b2147a6e34eade8743ba388fd688cfe61f51` (Qwen3.5 currently requires the vLLM main line)
- decoding: temperature 0, seed 0, max model length 32768, float16

This worker intentionally does not run native evaluation. Native SWE judges remain
a separate CPU/EBS execution concern.

After `/health` becomes ready, `jlens-runtime-identity.service` freezes SHA-256
receipts for every model shard, config, tokenizer, chat template, the resolved
container image ID, GPU UUID/driver, and `/v1/models` response under
`/opt/jlens-evidence/runtime`. Those receipt files are made read-only.

Once the worker is ready, open an SSH tunnel:

```sh
ssh -N -L 18000:127.0.0.1:8000 ubuntu@WORKER_IP
```

Then run `run-r075-calibration.sh REPO_ROOT`. It first replays the six exact
rendered prompts, then invokes the formal Round1 runner for the same three paired
tasks, collects its append-only attempts, and evaluates the frozen gate.
