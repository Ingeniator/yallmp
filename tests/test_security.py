from app.core.security import _redact_value, redact_headers


def test_redact_value_short():
    assert _redact_value("abc") == "[REDACTED]"
    assert _redact_value("abcd") == "[REDACTED]"


def test_redact_value_long():
    assert _redact_value("abcde") == "abcd...[REDACTED]"
    assert _redact_value("Bearer sk-12345") == "Bear...[REDACTED]"


def test_redact_headers_sensitive():
    headers = {
        "authorization": "Bearer secret-token",
        "cookie": "session=abc",
        "x-api-key": "key-12345",
        "x-token": "tok",
        "set-cookie": "id=xyz123",
        "proxy-authorization": "Basic creds123",
    }
    result = redact_headers(headers)
    for key in headers:
        assert "[REDACTED]" in result[key]


def test_redact_headers_preserves_non_sensitive():
    headers = {"content-type": "application/json", "accept": "text/html"}
    result = redact_headers(headers)
    assert result == headers


def test_redact_headers_does_not_mutate_original():
    original = {"authorization": "Bearer secret-token", "accept": "text/html"}
    copy = dict(original)
    redact_headers(original)
    assert original == copy
