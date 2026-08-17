#!/usr/bin/env python3
"""Minimal provider-neutral OpenAI completions worker for CUDA calibration."""

from __future__ import annotations

import argparse
import time
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForImageTextToText, AutoTokenizer


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    seed: int = 0
    stop: str | list[str] | None = None


def create_app(model_root: str, served_model_name: str) -> FastAPI:
    tokenizer = AutoTokenizer.from_pretrained(model_root, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_root,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="sdpa",
    ).eval()
    app = FastAPI()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": served_model_name}]}

    @app.post("/v1/completions")
    def completions(request: CompletionRequest) -> dict[str, Any]:
        if request.model != served_model_name:
            raise HTTPException(status_code=404, detail="unknown model")
        torch.manual_seed(request.seed)
        inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
        prompt_tokens = int(inputs.input_ids.shape[-1])
        sampling = request.temperature > 0
        generated = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            do_sample=sampling,
            temperature=request.temperature if sampling else None,
            top_p=request.top_p if sampling else None,
            pad_token_id=tokenizer.eos_token_id,
        )
        completion_ids = generated[0, prompt_tokens:]
        text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        stops = [request.stop] if isinstance(request.stop, str) else request.stop or []
        for stop in stops:
            position = text.find(stop)
            if position >= 0:
                text = text[:position]
        completion_tokens = int(completion_ids.shape[-1])
        return {
            "id": f"cmpl-{time.time_ns()}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": served_model_name,
            "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--served-model-name", default="same-base-cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(
        create_app(args.model_root, args.served_model_name),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
