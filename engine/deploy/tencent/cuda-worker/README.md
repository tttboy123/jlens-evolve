# Tencent Cloud CUDA execution worker

This bundle is the Tencent Cloud execution-plane adapter for the same generic
OpenAI-compatible model worker used by the local control plane. It does not add
Tencent, Qwen, or GPU-instance logic to the core loop.

Frozen resource for r075:

- instance: `ins-m6oi5jxo`, `GN7.2XLARGE32`, 1 x T4 16 GiB
- region/zone: `ap-hongkong` / `ap-hongkong-2`
- OS: Ubuntu Server 22.04 LTS, system disk 150 GiB Premium Cloud Disk
- network: SSH only from the operator `/32`; port 8000 is localhost-only
- base model: `Qwen/Qwen3.5-4B`; CUDA realization pinned to
  `cyankiwi/Qwen3.5-4B-AWQ-4bit@ef85d23bebaba87b3c4672ba11c449c79dbdb23e`
- runtime: pinned vLLM image, eager mode, 16,384-token limit, Triton GDN,
  persistent host cache; the quantized runtime must pass the MLX calibration
  gate before long runs

Bootstrap is delivered through Tencent Automation Tools (TAT) because the
operator Mac currently routes this public IP through a transparent proxy. TAT
requires no inbound inference port and is also used to retrieve append-only
runtime receipts. The server remains available only at `127.0.0.1:8000` on the
worker.

Cost guard for the current round:

- quoted compute ceiling: CNY 7.24/hour
- reviewed run budget: CNY 30; maximum runtime 4 hours; idle timeout 15 minutes
- provider-side action timer is the independent stop fallback
- every exit path must checkpoint, sync evidence, verify recovery, then stop
- instance termination is prohibited in the normal lifecycle; new workers must
  enable API termination protection and must not install a terminate action timer
- a stopped instance can still retain disk/public-IP/data-service charges;
  resource inventory is re-queried after every stop

Do not resume the 60-cell holdout until the frozen three-task MLX-vs-CUDA gate
passes. Native SWE evaluation remains a separate CPU/storage worker concern.

The r075 calibration passed 6/6 structural comparisons with zero safety
regressions. New runs must reuse the project-level worker when one exists and
pass `INSTANCE_ID`, `REGION`, and `ZONE` to this bootstrap. Do not reuse the FP16
Transformers fallback for long prompts: its SDPA attention OOMed at 9,173 prompt
tokens on a 16 GiB T4.
