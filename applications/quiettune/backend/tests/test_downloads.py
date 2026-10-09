import hashlib
import importlib
import json
import re

import pytest


@pytest.mark.parametrize("proxy", [None, "http://127.0.0.1:10809", "http://user:a'b@127.0.0.1:10809"],
                         ids=["直连", "本机代理", "代理参数引号"])
def test_windows_download_proxy_and_manifest(tmp_path, monkeypatch, proxy):
    """只验证下载协议及缓存清单，不下载或替换正式模型。"""
    module = importlib.import_module("backend.download_models")
    monkeypatch.setattr(module, "MODELS", tmp_path)
    monkeypatch.setattr(module.platform, "system", lambda: "Windows")
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    if proxy:
        monkeypatch.setenv("HTTPS_PROXY", proxy)
    commands = []
    payload = b"download-protocol-test"

    def request(command, check, timeout):
        assert check and timeout == 600
        script = command[-1]
        commands.append(script)
        filename = re.search(r"-OutFile '((?:''|[^'])*)'", script).group(1).replace("''", "'")
        from pathlib import Path
        Path(filename).write_bytes(payload)

    monkeypatch.setattr(module.subprocess, "run", request)
    module.download()
    assert len(commands) == 6
    if proxy:
        assert all("-Proxy '" + proxy.replace("'", "''") + "'" in script for script in commands)
    else:
        assert all("-Proxy" not in script for script in commands)
    manifest = json.loads((tmp_path / "manifest.json").read_text("utf-8"))
    assert len(manifest["files"]) == 6
    assert all(file["sha256"] == hashlib.sha256(payload).hexdigest() for file in manifest["files"])
    assert "CC BY-NC-SA 4.0" in manifest["license"] and "CC BY-NC-ND 4.0" in manifest["license"]
    assert manifest["license_source"] == "https://essentia.upf.edu/models/LICENSE"
    assert len(manifest["license_references"]) == 2
