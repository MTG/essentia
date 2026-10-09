"""缓存官方权重与原始元数据，保留来源、版本和本地校验信息。"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import urllib.request
from datetime import datetime, timezone

from backend.app.config import MODELS
from backend.app.models import MODEL_SPECS


def download():
    manifest = {"downloaded_at": datetime.now(timezone.utc).isoformat(),
                "license": "官方声明存在 CC BY-NC-SA 4.0 与 CC BY-NC-ND 4.0 差异，请以权利方确认结果为准。",
                "license_source": "https://essentia.upf.edu/models/LICENSE",
                "license_references": ["https://essentia.upf.edu/models.html", "https://essentia.upf.edu/licensing_information.html"],
                "checksum_note": "SHA-256 为下载后计算，用于本地完整性核对；官方未在这些 JSON 中发布摘要。", "files": []}
    for name, directory in MODEL_SPECS.items():
        for extension in ("json", "pb"):
            url = f"https://essentia.upf.edu/models/{directory}/{name}.{extension}"
            target = MODELS / f"{name}.{extension}"
            temporary = target.with_suffix(target.suffix + ".part")
            print(f"下载官方模型资源：{name}.{extension}", flush=True)
            if platform.system() == "Windows":
                # Windows 系统证书可兼容使用组织代理证书的本地网络。
                script = f"$ErrorActionPreference='Stop'; Invoke-WebRequest -Uri '{url}' -OutFile '{str(temporary).replace(chr(39), chr(39)*2)}'"
                proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
                if proxy:
                    script += " -Proxy '" + proxy.replace("'", "''") + "'"
                subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True, timeout=600)
            else:
                with urllib.request.urlopen(url, timeout=180) as response, temporary.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
            temporary.replace(target)
            manifest["files"].append({"filename": target.name, "source": url, "bytes": target.stat().st_size,
                                      "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    (MODELS / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
    print("模型已下载，分析歌曲时将执行真实推理。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载 QuietTune 使用的官方非商业音乐模型。")
    parser.parse_args()
    download()
