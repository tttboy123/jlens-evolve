from __future__ import annotations

import pytest

from loopback_connect_proxy import ProxyProtocolError, is_allowed_target, parse_connect


def test_proxy_allowlist_is_limited_to_required_service_domains():
    assert is_allowed_target("api.openai.com", 443) is True
    assert is_allowed_target("chatgpt.com", 443) is True
    assert is_allowed_target("cdn.oaistatic.com", 443) is True
    assert is_allowed_target("files.oaiusercontent.com", 443) is True
    assert is_allowed_target("registry-1.docker.io", 443) is True
    assert is_allowed_target("auth.docker.io", 443) is True
    assert is_allowed_target("production.cloudflare.docker.com", 443) is True
    assert is_allowed_target("github.com", 443) is True
    assert is_allowed_target("codeload.github.com", 443) is True
    assert is_allowed_target("raw.githubusercontent.com", 443) is True
    assert is_allowed_target("proxy.golang.org", 443) is True
    assert is_allowed_target("sum.golang.org", 443) is True
    assert is_allowed_target("storage.googleapis.com", 443) is True
    assert is_allowed_target("dl-ssl.google.com", 443) is True
    assert is_allowed_target("dl.google.com", 443) is True
    assert is_allowed_target("deb.nodesource.com", 443) is True
    assert is_allowed_target("registry.npmjs.org", 443) is True
    assert is_allowed_target("github.example.com", 443) is False
    assert is_allowed_target("127.0.0.1", 443) is False
    assert is_allowed_target("api.openai.com", 80) is False


def test_proxy_parses_connect_and_rejects_plain_http():
    assert parse_connect(b"CONNECT api.openai.com:443 HTTP/1.1\r\nHost: x\r\n\r\n") == (
        "api.openai.com",
        443,
    )
    with pytest.raises(ProxyProtocolError, match="CONNECT"):
        parse_connect(b"GET https://api.openai.com/ HTTP/1.1\r\n\r\n")
