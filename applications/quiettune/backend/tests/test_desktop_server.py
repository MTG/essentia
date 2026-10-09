import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest


@pytest.mark.parametrize("shutdown", [True, False], ids=["关闭指令", "父进程管道关闭"])
def test_desktop_service_lifecycle(tmp_path, shutdown):
    """随机端口健康检查后，两种退出路径都必须释放端口。"""
    root = Path(__file__).resolve().parents[2]
    environment = {**os.environ, "QUIETTUNE_DATA": str(tmp_path / "desktop-data"), "PYTHONUTF8": "1"}
    log = (tmp_path / "service.log").open("w", encoding="utf-8")
    process = subprocess.Popen([sys.executable, "-u", "-m", "backend.desktop_server"], cwd=root,
                               env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                               text=True, encoding="utf-8")
    try:
        message = json.loads(process.stdout.readline())
        assert message["type"] == "quiettune-service"
        assert 0 < message["port"] <= 65535
        url = f"http://127.0.0.1:{message['port']}"
        ready = False
        deadline = time.monotonic() + 30
        with httpx.Client(timeout=1, trust_env=False) as client:
            while time.monotonic() < deadline:
                try:
                    ready = client.get(url + "/api/health").json()["status"] == "ok"
                    if ready:
                        break
                except (httpx.HTTPError, ValueError):
                    time.sleep(0.2)
        assert ready, "桌面服务没有就绪。"
        if shutdown:
            process.stdin.write("shutdown\n")
            process.stdin.flush()
        else:
            process.stdin.close()
        assert process.wait(timeout=15) == 0
        with pytest.raises((httpx.ConnectError, httpx.ConnectTimeout)):
            httpx.get(url + "/api/health", timeout=1, trust_env=False)
    finally:
        if not process.stdin.closed:
            process.stdin.close()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        log.close()
