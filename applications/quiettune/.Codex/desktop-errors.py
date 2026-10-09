"""验证桌面启动错误、重试、单实例与服务异常退出后的恢复。"""

import json
import os
import socket
import subprocess
import time
from pathlib import Path

import httpx
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".Codex/desktop-test" / f"recovery-{time.time_ns()}"
WORK.mkdir(parents=True)
EXE = Path(os.environ.get("QUIETTUNE_TEST_EXE", ROOT / ".Codex/releases/win-unpacked/QuietTune.exe"))
blocked = WORK / "data"
blocked.write_text("用于验证启动失败的普通文件。", "utf-8")
with socket.socket() as listener:
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
environment = {**os.environ, "QUIETTUNE_DESKTOP_HOME": str(WORK / "home"), "QUIETTUNE_DATA": str(blocked),
               "PYTHONUTF8": "1", "PATH": os.pathsep.join([str(Path(os.environ["WINDIR"]) / "System32"), os.environ["WINDIR"]])}
for name in ("QUIETTUNE_MODELS", "PYTHONPATH", "PYTHONHOME", "ELECTRON_RUN_AS_NODE"):
    environment.pop(name, None)
output = (WORK / "launch.log").open("w", encoding="utf-8")
process = subprocess.Popen([str(EXE), "--quiettune-test", f"--remote-debugging-port={port}"],
                           env=environment, cwd=ROOT, stdout=output, stderr=output)
try:
    with sync_playwright() as playwright:
        with httpx.Client(timeout=1, trust_env=False) as client:
            for _ in range(150):
                try:
                    if client.get(f"http://127.0.0.1:{port}/json/version").status_code == 200:
                        break
                except httpx.HTTPError:
                    time.sleep(0.2)
            else:
                raise AssertionError("未能连接 Electron。")
        browser = playwright.chromium.connect_over_cdp(f"http://127.0.0.1:{port}")
        page = browser.contexts[0].pages[0]
        page.get_by_role("button", name="重新启动", exact=True).wait_for(timeout=60000)
        assert "失败" in page.locator("#message").inner_text()
        assert page.get_by_role("button", name="查看日志", exact=True).is_visible()
        blocked.unlink()
        page.get_by_role("button", name="重新启动", exact=True).click()
        page.get_by_role("heading", name="音乐舒适度总览").wait_for(timeout=60000)
        url = page.url.rstrip("/")
        second = subprocess.Popen([str(EXE), "--quiettune-test"], env=environment, cwd=ROOT, stdout=output, stderr=output)
        assert second.wait(timeout=15) == 0, "重复打开没有正确复用实例。"
        assert len(browser.contexts[0].pages) == 1

        # 只结束此测试 Electron 创建的 Python 服务，验证真实意外退出而非模拟页面状态。
        powershell = Path(os.environ["WINDIR"]) / "System32/WindowsPowerShell/v1.0/powershell.exe"
        command = f"Get-CimInstance Win32_Process -Filter 'ParentProcessId = {process.pid}' | Where-Object Name -eq 'python.exe' | Select-Object -ExpandProperty ProcessId"
        identity = subprocess.check_output([str(powershell), "-NoProfile", "-Command", command], text=True).strip()
        assert identity.isdigit(), "未找到测试所属的分析服务。"
        subprocess.run([str(Path(os.environ["WINDIR"]) / "System32/taskkill.exe"), "/PID", identity, "/F"],
                       check=True, stdout=output, stderr=output)
        page.get_by_role("button", name="重新启动", exact=True).wait_for(timeout=20000)
        assert "意外退出" in page.locator("#message").inner_text()
        page.get_by_role("button", name="重新启动", exact=True).click()
        page.get_by_role("heading", name="音乐舒适度总览").wait_for(timeout=60000)
        with httpx.Client(timeout=5, trust_env=False) as client:
            assert client.get(page.url.rstrip("/") + "/api/stats").json()["total"] == 0
        page.close()
        assert process.wait(timeout=25) == 0
        result = {"status": "通过", "checked": ["实际启动失败页", "日志入口", "修复后重试", "单实例", "服务意外退出提示", "重新启动恢复"]}
        (ROOT / ".Codex/desktop-errors-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
finally:
    if process.poll() is None:
        subprocess.run([str(Path(os.environ["WINDIR"]) / "System32/taskkill.exe"), "/PID", str(process.pid), "/T", "/F"],
                       stdout=output, stderr=output)
        process.wait(timeout=30)
    output.close()
print("桌面启动失败、重试、单实例与异常服务恢复验证通过。")
