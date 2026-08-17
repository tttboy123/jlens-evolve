"""Authorized, replay-safe boundary around an injected parent-model transport."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .contracts import (
    ContractError,
    LoopAuthorization,
    ParentModelRequest,
    ParentModelResponse,
)
from .ledger import ParentCallLedger

ParentTransport = Callable[[ParentModelRequest], dict[str, Any]]


class ParentModelAdapter:
    """Validate, reserve, dispatch once, and freeze a parent-model response."""

    def __init__(self, *, ledger: ParentCallLedger, transport: ParentTransport) -> None:
        self.ledger = ledger
        self.transport = transport

    def generate(
        self,
        *,
        call_id: str,
        request: ParentModelRequest,
        authorization: LoopAuthorization,
    ) -> ParentModelResponse:
        request.validate()
        authorization.validate()
        authorization.assert_active()
        if authorization != self.ledger.authorization:
            raise ContractError("authorization does not match the parent call ledger")

        existing = self.ledger.get(call_id)
        if existing is not None:
            if existing.request_sha256 != request.sha256:
                raise ContractError("call_id already belongs to a different request")
            if existing.status == "completed" and existing.response is not None:
                return ParentModelResponse.from_dict(existing.response)
            raise ContractError(
                "parent call was already reserved and cannot be dispatched again"
            )

        self.ledger.reserve(call_id=call_id, request_sha256=request.sha256)
        try:
            raw = self.transport(request)
            response = ParentModelResponse.from_dict(raw)
        except Exception as exc:
            self.ledger.abort(call_id=call_id, reason=f"{type(exc).__name__}: {exc}")
            raise
        self.ledger.complete(
            call_id=call_id,
            response_sha256=response.sha256,
            response=response.to_dict(),
            usage=response.usage,
        )
        return response
