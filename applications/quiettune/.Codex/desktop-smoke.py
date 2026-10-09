"""验证真正的 Electron 发行目录，使用独立数据与移除开发工具的 PATH。"""

import json
import os
import socket
import subprocess
import time
from pathlib import Path

import httpx
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".Codex" / "desktop-test"
WORK.mkdir(parents=True, exist_ok=True)
EXE = Path(os.environ.get("QUIETTUNE_TEST_EXE", ROOT / ".Codex/releases/win-unpacked/QuietTune.exe"))
HOME = WORK / f"home-{time.time_ns()}"
screens = ROOT / ".Codex/screenshots"
screens.mkdir(exist_ok=True)
errors = []


def launch(playwright):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    environment = {**os.environ, "QUIETTUNE_DESKTOP_HOME": str(HOME),
                   "QUIETTUNE_DATA": str(HOME / "data"), "PYTHONUTF8": "1",
                   "PATH": os.pathsep.join([str(Path(os.environ["WINDIR"]) / "System32"), os.environ["WINDIR"]])}
    for name in ("QUIETTUNE_MODELS", "QUIETTUNE_NODE", "QUIETTUNE_FFMPEG", "QUIETTUNE_FFPROBE", "PYTHONPATH", "PYTHONHOME", "ELECTRON_RUN_AS_NODE"):
        environment.pop(name, None)
    output = (WORK / "launch.log").open("a", encoding="utf-8")
    process = subprocess.Popen([str(EXE), "--quiettune-test", f"--remote-debugging-port={port}"],
                               cwd=ROOT, env=environment, stdout=output, stderr=output)
    deadline = time.monotonic() + 120
    with httpx.Client(timeout=1, trust_env=False) as client:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise AssertionError("Electron 提前退出，请查看桌面测试日志。")
            try:
                if client.get(f"http://127.0.0.1:{port}/json/version").status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
        else:
            raise AssertionError("Electron 调试连接没有就绪。")
    browser = playwright.chromium.connect_over_cdp(f"http://127.0.0.1:{port}")
    page = browser.contexts[0].pages[0]
    page.on("pageerror", lambda error: errors.append(str(error)))
    try:
        page.get_by_role("heading", name="音乐舒适度总览").wait_for(timeout=120000)
    except Exception:
        page.screenshot(path=str(screens / "electron-startup-failure.png"))
        page.close()
        process.wait(timeout=25)
        output.close()
        raise
    return process, browser, page, output


def close(process, page, output, url):
    page.close()
    assert process.wait(timeout=25) == 0, "Electron 未正常退出。"
    output.close()
    with httpx.Client(timeout=1, trust_env=False) as client:
        try:
            client.get(url + "/api/health")
        except (httpx.ConnectError, httpx.ConnectTimeout):
            return
    raise AssertionError("窗口关闭后分析服务仍在运行。")


with sync_playwright() as playwright:
    process, browser, page, output = launch(playwright)
    url = page.url.rstrip("/")
    try:
        assert ":8000" not in url, "桌面服务占用了已有 Web 端口。"
        with httpx.Client(base_url=url, timeout=30, trust_env=False) as client:
            health = client.get("/api/health").json()
            assert client.get("/api/profile").json()["name"] == "个人偏好"
            assert page.get_by_role("button", name="个人听觉偏好").is_visible()
            assert health["ffmpeg"] and health["models"]["status"] == "available", health
            assert Path(health["model_path"]).is_relative_to(HOME), health
            page.locator('input[type="file"]').first.set_input_files(str(ROOT / ".Codex/fixtures/Vibe-Ace.ogg"))
            deadline = time.monotonic() + 240
            detail = None
            while time.monotonic() < deadline:
                tracks = client.get("/api/tracks").json()
                if tracks:
                    detail = client.get(f"/api/tracks/{tracks[0]['id']}").json()
                    if detail["status"] in {"completed", "failed"}:
                        break
                time.sleep(0.5)
            assert detail and detail["status"] == "completed", detail
            assert detail["features"]["ai"]["status"] == "ready", detail["features"]["ai"]
            assert len(detail["score"]["segments"]) == 13
            identity = detail["id"]
            analyzed_at = detail["analyzed_at"]
            ranged = client.get(f"/api/tracks/{identity}/audio", headers={"Range": "bytes=100-299"})
            assert ranged.status_code == 206 and len(ranged.content) == 200
            page.get_by_role("button", name="音乐库", exact=False).first.click()
            page.get_by_role("textbox", name="搜索歌曲").fill("不会匹配的测试标题")
            assert page.get_by_text("没有符合条件的歌曲，试试放宽筛选条件。").is_visible()
            page.get_by_role("textbox", name="搜索歌曲").fill("")
            page.get_by_role("button", name="Vibe Ace", exact=False).first.click()
            page.get_by_role("heading", name="听见每个片段的变化").wait_for()
            page.get_by_role("button", name="试听最高片段", exact=False).click()
            page.wait_for_timeout(1200)
            playback = page.locator("audio").evaluate("audio => ({time:audio.currentTime,paused:audio.paused,ready:audio.readyState})")
            assert playback["ready"] >= 2 and playback["time"] > 0 and not playback["paused"], playback
            page.get_by_role("button", name="一般", exact=True).click()
            page.get_by_role("button", name="保存我的评价").click()
            page.get_by_text("评价已保存。", exact=True).wait_for()
            assert client.get(f"/api/tracks/{identity}").json()["analyzed_at"] == analyzed_at
            page.screenshot(path=str(screens / "electron-detail.png"), full_page=True)
            page.get_by_role("button", name="暂停", exact=True).click()
            page.get_by_role("button", name="切换浅色主题").click()
            assert page.locator("html").get_attribute("data-theme") == "light"
            page.get_by_role("button", name="切换深色主题").click()
            page.get_by_role("button", name="偏好校准", exact=True).click()
            page.get_by_role("button", name="训练个人模型").click()
            page.get_by_text("至少需要 12 首已分析歌曲和 3 种不同评价", exact=False).first.wait_for()
            exported = client.get("/api/export.csv")
            assert exported.status_code == 200 and "Vibe Ace" in exported.text
            assert "个人指数" in exported.text.splitlines()[0]
            (WORK / "导出验证.csv").write_bytes(exported.content)
            page.get_by_role("button", name="总览", exact=True).click()
            page.screenshot(path=str(screens / "electron-dashboard.png"), full_page=True)
            page.get_by_role("button", name="切换浅色主题").click()
            volume = page.get_by_role("slider", name="播放音量")
            volume.press("Home")
            volume.press("ArrowRight")
            page.wait_for_timeout(300)
            result = {"status": "通过", "executable": str(EXE), "backend_url": url, "without_system_developer_tools": True,
                      "health": health, "ai": detail["features"]["ai"], "score": detail["score"]["combined"], "playback": playback,
                      "checked": ["真实 Electron", "模型缓存", "界面导入真实音频", "实际 AI 推理", "分段图表", "搜索", "Range", "时间跳转播放", "反馈缓存", "深浅主题", "偏好不足提示", "CSV 导出", "重启持久化", "关闭服务清理"]}
    finally:
        close(process, page, output, url)
    process, browser, page, output = launch(playwright)
    url = page.url.rstrip("/")
    try:
        with httpx.Client(base_url=url, timeout=10, trust_env=False) as client:
            restored = client.get(f"/api/tracks/{identity}").json()
            assert restored["feedback"]["rating"] == 3 and restored["analyzed_at"] == analyzed_at
        assert page.locator("html").get_attribute("data-theme") == "light", "重启后主题丢失。"
        assert float(page.get_by_role("slider", name="播放音量").input_value()) == 0.01, "重启后音量丢失。"
        assert not errors, errors
        result["console_errors"] = errors
        result["feedback_persisted"] = True
        result["theme_and_volume_persisted"] = True
    finally:
        close(process, page, output, url)
    (ROOT / ".Codex/desktop-smoke-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
print("完整 Windows 桌面包、真实模型、界面播放、数据持久化与服务清理验证通过。")
