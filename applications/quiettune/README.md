# QuietTune — 音乐舒适度分析系统

面向 Windows 11 的 **Electron + React 桌面应用**：真实声学特征、官方音乐情绪模型、分段试听和个人偏好学习。Electron 管理中文窗口和私有 Python 分析服务，音频、结果和反馈都保存在本机。

## 源码与许可

应用位于 [JadeCake5/essentia 的 develop 分支](https://github.com/JadeCake5/essentia/tree/develop/applications/quiettune) 的 `applications/quiettune/`。fork 保留 [MTG/essentia](https://github.com/MTG/essentia) 的原有核心与历史，QuietTune 在独立目录中调用官方库；从源码运行应用不要求编译上游 C++ 工程。

新增应用代码采用 GNU AGPL v3，原文见 [LICENSE](LICENSE)。第三方库、模型和其他资源的范围见 [LICENSE-NOTICES.md](LICENSE-NOTICES.md)。仓库包含完整应用源码、测试、依赖锁和构建配置，不包含用户音乐、数据库、模型权重、运行环境或安装包。

使用 Git 获取制作分支，可以只检出应用目录：

```powershell
git clone --filter=blob:none --sparse --branch develop https://github.com/JadeCake5/essentia.git
cd essentia
git sparse-checkout set applications/quiettune
cd applications/quiettune
```

下文的项目目录均指 `applications/quiettune/`，其中的 `.Codex/` 保存本地构建、缓存和验证结果。

## Windows 启动

### 使用完整桌面包

完成下文的源码准备和桌面构建后，双击 `.Codex/releases/QuietTune-1.1.2-Windows-x64-安装.exe`。中文安装器可选择安装目录并创建快捷方式。也可直接运行 `.Codex/releases/win-unpacked/QuietTune.exe`；使用免安装目录时须保留整个目录，不能只复制一个 EXE。GitHub 源码仓库不附带预先构建的安装包。

窗口顶部与界面融为一体，保留系统的最小化、最大化和关闭按钮；拖动顶部空白区域可以移动窗口。顶部“应用菜单”按钮提供文件、编辑、视图、帮助功能，窗口按钮色彩跟随深浅主题切换，启动页和错误页也可拖动。

桌面包内含 Python 3.11.6、全部分析依赖、Node 22.18.0、FFmpeg/FFprobe、官方 WASM 与三个真实模型。运行时无需另装这些工具，也无需 Docker。首次加载科学计算库及首次分析需要稍等；启动页显示状态，失败时可查看日志和重试。

正式桌面版数据位于 `%APPDATA%/QuietTune/data`，模型缓存位于 `%APPDATA%/QuietTune/models`，日志位于 `%APPDATA%/QuietTune/logs`。菜单“文件”可打开数据和模型目录，“帮助”可打开日志及许可。模型在首次启动时从包内复制，不从网络下载；离线也可分析。

部分打包应用或沙箱宿主会重定向 Windows 用户目录。请以“分析设置”显示的实际路径为准，或直接使用菜单打开数据目录。

应用自动分配独立本机端口，兼容同时运行原 Web 服务；退出窗口时结束自己启动的分析服务。正在分析的任务在关闭超过 12 秒后会被中断，下次启动恢复。主题、播放音量、窗口大小与位置跨重启保存。连续打开两次只显示同一实例。

卸载保留音乐库、评价与模型缓存；数据备份可在关闭应用后复制完整 `%APPDATA%/QuietTune`。免安装版也使用该用户数据目录，删除程序目录不会删除音乐库。

官方模型有非商业使用限制，官网 SA/ND 许可声明存在差异，详见 [LICENSE-NOTICES.md](LICENSE-NOTICES.md)。本次仅发布源码，模型需从官方来源自行下载。桌面构建未配置发行者代码签名。

### 从源代码启动桌面版

需要 Python 3.11、Node.js 22，以及 PATH 中可用的 FFmpeg/FFprobe。

在项目目录运行：

```powershell
./start.cmd -Install
```

首次启动安装项目专属依赖、构建界面、下载三个官方模型，然后打开 Electron 窗口。再次启动：

```powershell
./start.cmd
```

源代码模式继续使用项目的 `data/` 和 `models/`，正式包默认使用独立用户目录。需要继续用浏览器时运行 `./start.cmd -Web`，打开 http://127.0.0.1:8000；端口被占用时使用 `./start.cmd -Web -Port 8001`。

`start.cmd` 用 UTF-8 读取中文脚本，兼容 Windows PowerShell 5。PowerShell 7 可以直接运行 `./start.ps1`。它们都不会安装系统后台服务。

需要通过本机代理下载时，在当前终端设置：

```powershell
$env:HTTP_PROXY = 'http://127.0.0.1:10809'
$env:HTTPS_PROXY = $env:HTTP_PROXY
$env:NO_PROXY = '127.0.0.1,localhost'
./start.cmd -Install
```

代理仅用于当前进程和后续下载，本机应用接口直连。Electron 官方下载器使用 EnvHttpProxyAgent；Python 运行时准备脚本也支持 `-Proxy`。

尚未安装 FFmpeg 的电脑可使用 `winget install Gyan.FFmpeg`，安装后重新打开终端。普通上传支持 MP3、FLAC、WAV、M4A、OGG，单文件不超过 200 MB，时长不超过 20 分钟，目前支持单声道/立体声。

## 日常使用

1. 在音乐库导入文件，或选择文件夹。浏览器只读取你明确选中的文件，保存相对路径信息，不扫描电脑。
2. 后台自动完成解码、声学分析和模型推理。音乐库显示阶段进度，失败可重新分析。
3. 单曲详情显示波形、5 秒片段曲线、短时响度、频段能量和情绪概率。点击片段即可跳转试听。
4. 保存舒适度评价，可随时修改。评价因素与备注保留供回看，当前回归训练使用五级总体评价。
5. 至少评价 12 首已分析歌曲、覆盖至少 3 种评价后，点击训练个人模型。模型通过四折验证且优于折内均值基线时才启用。点击右上角切换个人指数；模型不足或失效时明确回退通用指数。
6. 调整权重、阈值会即时对缓存特征重新评分；改变窗口或 AI 状态后，可选择重新分析音乐库。

首次启动音乐库为空，需要导入自己的音乐。本地验证使用 Kevin MacLeod 的《Vibe Ace》，其署名随验证工具提供，音频不随仓库上传。

## 实际分析算法

- FFmpeg 保留原文件的响度测量通道，使用 EBU R128 测量整曲 LUFS、3 秒短时响度与真峰值；数字静音或不足以测量的指标返回空值。
- 单声道 22050 Hz 辅助分析，STFT 2048、步长 512；提取 RMS、采样峰值、峰均比、频谱质心、85% 滚降、平坦度、4 kHz 以上能量、四频段能量。
- librosa 起始包络检测结合最小绝对包络强度与归一化谱变化，避免纯音中数值噪声被归一化为大量假事件。瞬态强度取归一化频谱的正向变化，减少母带增益影响。
- 片段为不重叠窗口，尾部不足窗口的真实时长仍保存。局部模型概率不被伪造：时间曲线仅是声学指数。
- 数字静音的声学评分为 0。静音的模型预测不用于综合评分。
- 频段分析受 22050 Hz 分析采样率限制，最高频段约到 11 kHz；质心与平坦度只是听感代理，并不能识别具体乐器或证明刺耳。

默认归一化参考：响度 −35 至 −8 LUFS/dBFS；质心 600 至 5000 Hz；高频占比 0.02 至 0.5；瞬态密度 0 至 6 次/秒；谱变化强度 0 至 0.2；平坦度 0.01 至 0.45。超出参考区间截断到 0–1。高频代理为 55% 高频能量与 45% 质心；瞬态代理为 75% 密度与 25% 强度。

整曲权重：激烈概率 22%、放松反向指标 15%、响度 8%、高频 23%、瞬态 22%、噪声状频谱 10%。AI 不可用时对实际可用维度重新归一化。

综合指数 = 整曲综合均值 + 25% × max(0，最高声学片段指数 − 整曲综合均值)。这些是应用内可修改的启发式参数，未经过人类听感数据的科学校准。BPM 不直接进入评分。

## 官方模型与兼容方案

使用官方 `msd-musicnn-1` 的 200 维嵌入，以及独立 `mood_aggressive-msd-musicnn-1`、`mood_relaxed-msd-musicnn-1` 分类头。

| 协议 | 经元数据与实际运行确认的值 |
| --- | --- |
| 模型音频输入 | 16000 Hz 单声道 |
| 梅尔输入 | 512 采样帧、256 步长、96 个梅尔频带 |
| MusiCNN 张量 | 批次 × 187 × 96 |
| 嵌入输出 | `model/dense/BiasAdd`，200 维 |
| 分类头输入/输出 | `model/Placeholder`，200 维 / `model/Softmax`，2 维 |
| 激烈类别 | `aggressive`、`not_aggressive`，按名称定位 |
| 放松类别 | `non_relaxed`、`relaxed`，按名称定位 |
| 模型版本 | 嵌入元数据版本 1；两个分类头元数据版本 2，尽管文件名后缀为 1 |
| 尾部处理 | patch_hop 187，尾部 repeat；短音频重复输入梅尔帧，不伪造预测 |

Windows 官方 Essentia Python wheel 不可用。本项目使用官方 `essentia.js 0.1.3` 的 WebAssembly `TensorflowInputMusiCNN`，不是自行近似梅尔谱；Python TensorFlow 2.15.1 的 GraphDef 兼容接口运行原始 `.pb`。本机 Python 3.11.6、NumPy 1.26.4、TensorFlow 2.15.1 的组合已完成真实推理。

Linux/Docker 使用 `essentia-tensorflow==2.1b6.dev1389` 的 Python `TensorflowInputMusiCNN` 与同一 GraphDef 推理。PyPI 已核对该版本提供 CPython 3.11 manylinux x86_64 wheel；本机没有 Docker/可用 WSL，Linux 路径尚未实际运行验证，不能将它视为已验证平台。

独立下载模型：

```powershell
./.venv/Scripts/python.exe -m backend.download_models
```

模型缺失、运行依赖异常或推理失败时，返回空概率与实际原因，并显示“仅声学特征评分”。`GET /api/health` 可查看缓存状态；权重已缓存不等于推理一定成功。模型不随代码提交；许可与来源见 [LICENSE-NOTICES.md](LICENSE-NOTICES.md)。

## Docker Compose

安装 Docker Desktop，启用 WSL2/虚拟化后，在项目根目录：

```powershell
docker compose build
docker compose run --rm quiettune python -m backend.download_models
docker compose up -d
```

已有项目模型缓存时可直接 `docker compose up --build -d`。WebUI 与 API 同源，地址为 http://127.0.0.1:8000 。使用 `docker compose logs -f quiettune` 查看日志，`docker compose down` 停止，绑定目录中的音乐与数据库保留。

Dockerfile 提供 Node 构建阶段及 Python/FFmpeg 运行阶段。Compose 仅绑定本机端口；无需 Redis、第三方云 API 或独立数据库服务。该配置已静态检查，但当前主机无法执行 Docker，因此构建、Linux wheel 加载及容器内模型推理尚待具备 Docker 的环境验证。

## 开发模式

桌面窗口复用正式 React 产物：

```powershell
npm.cmd start
```

修改前端后运行 `npm.cmd run build:ui` 再重新加载桌面窗口。即时前端开发继续使用以下 Vite 与 FastAPI 组合。

后端：

```powershell
./.venv/Scripts/python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

前端另一终端：

```powershell
npm.cmd run dev --prefix frontend
```

打开 http://127.0.0.1:5173 。Vite 将 `/api` 代理到 8000；正式构建由 FastAPI 同源提供，无需额外代理服务器。

## 构建 Windows 桌面包

首次按源代码启动步骤安装依赖并下载模型后：

```powershell
npm.cmd run build:ui
npm.cmd run prepare:runtime
npm.cmd run pack
npm.cmd run dist
```

`prepare:runtime` 下载官方嵌入式 Python，复制已锁定的项目依赖和本机 Node/FFmpeg，保留第三方许可并实际加载 TensorFlow 验证。`pack` 生成完整免安装目录；`dist` 生成中文 NSIS 安装器，两者都会先检查必要资源。Windows x64 是当前桌面发行目标，不把未验证的 macOS/Linux 桌面包列为支持平台。

构建产物和中间运行环境位于 `.Codex/`。TensorFlow 及科学计算库占用较大空间，选择目录式运行环境避免每次启动解压。运行环境清单保留来源、版本、摘要和完整 Python 依赖锁。根目录 `package-lock.json` 锁定 Electron、打包器和下载代理依赖。

## 目录与数据

```text
backend/app/          接口、数据库、评分、任务与真实音频/模型分析
backend/tests/        pytest 单元与 API 测试
backend/download_models.py  官方模型下载与来源清单
frontend/src/         中文 React 界面、播放器与图表
desktop/              Electron 主进程、启动页、偏好桥接与标准打包配置
backend/desktop_server.py  随机端口服务与桌面进程退出协议
data/audio/           内容哈希命名的导入副本
data/covers/          元数据中提取的真实封面
data/quiettune.db     SQLite 特征、片段、任务、反馈、偏好与评分版本
models/              官方权重、原始 JSON 与 SHA-256 来源清单
.Codex/              公开验证工具，以及本机生成且不提交的缓存与结果
```

歌曲文件不嵌入数据库；数据库仅保存相对于音乐目录的文件名，因此复制完整数据目录后可在 Windows 与 Linux 路径间迁移。SQLAlchemy 外键保证删除歌曲时清理对应特征、任务与反馈；正在分析的歌曲需任务结束后删除。

后台最多 4 个线程，实际并发可在设置中调整；AI 推理串行并以最多 16 个输入片段一批运行。任务配置在排队时快照，重启会恢复未完成任务。单机部署只启动 **一个 Uvicorn 进程**，不要使用多 worker 启动多个队列。

备份或迁移：先正常停止服务，复制完整 `data/` 与 `models/`，然后设置环境变量 `QUIETTUNE_DATA`、`QUIETTUNE_MODELS` 重启。不要只在运行中复制 `.db` 而忽略 SQLite 写前日志。评分权重变化、反馈修改、歌曲删除或特征重算会使个人模型失效，需重新训练；音频特征仍可复用。

数据库初始化包含历史偏好标识的精确迁移，用户歌曲标题、备注和训练参数保留。后续结构变更应提供明确迁移。数据回滚前应停止服务并使用完整备份目录。

## 本地验证

```powershell
./.venv/Scripts/python.exe -m pytest backend/tests -q
npm.cmd run build --prefix frontend
```

pytest 使用自动生成的独立测试数据，其中真实模型测试需要先完成官方模型下载。桌面验收还需要已经构建的免安装目录和 Python 的 Playwright、httpx、Pillow；可以在独立验证虚拟环境中安装这些工具：

```powershell
python -m venv .Codex/verification-tools
./.Codex/verification-tools/Scripts/python.exe -m pip install playwright httpx pillow
$env:PATH = (Resolve-Path .Codex/verification-tools/Scripts).Path + ';' + $env:PATH
New-Item -ItemType Directory -Force .Codex/fixtures | Out-Null
Invoke-WebRequest -Uri 'https://librosa.org/data/audio/Kevin_MacLeod_-_Vibe_Ace.ogg' -OutFile '.Codex/fixtures/Vibe-Ace.ogg'
python .Codex/desktop-smoke.py
python .Codex/desktop-errors.py
node .Codex/desktop-titlebar.cjs
node .Codex/check-titlebar-release.cjs
```

需要代理时，音频下载的 Invoke-WebRequest 同样添加 `-Proxy $env:HTTPS_PROXY`。音频署名见 [.Codex/fixtures/Vibe-Ace-license.txt](.Codex/fixtures/Vibe-Ace-license.txt)。桌面脚本自行打开真正 Electron 包，使用独立测试数据，验证真实模型、播放、偏好持久化、标题栏和关闭清理，不重新安装用户应用。

Windows 完整 Python 依赖锁为 `backend/requirements-windows.lock`；Linux 完整依赖锁为 `backend/requirements-linux.lock`。跨平台直接依赖为 `backend/requirements.txt`，npm 使用各目录 `package-lock.json`。

自动测试覆盖静音、增益不变性、高频、瞬态、分段尾部、LUFS、真实模型契约与推理、上传去重、Range、反馈修改、缓存重评分、个人模型验证和排序、后台失败恢复与删除一致性。合成音频的情绪预测只能证明推理路径运行，不是分类准确率证明。

验证脚本将结果和截图写入本机 `.Codex/`，这些生成文件不纳入公开源码。Windows 本地已完成 26 项 pytest、真实模型与桌面流程验证；Linux/Docker 尚未运行验证。

## 当前限制

- 个人模型需要真实试听评价；标注不足或验证未通过时使用通用评分。
- 声学代理、固定权重与模型概率仍需真实听众反馈校准；不推断具体乐器、尖锐人声或医学舒适度。
- 完整音频上限 20 分钟；分析期间全曲 STFT 与梅尔特征占用内存，普通电脑建议单任务。
- Docker/Linux 尚未实测，Windows 已验证路径是当前可运行交付。
- 浏览器原生播放能力依格式而异，常见 MP3/WAV/OGG 可直接试听；特殊编码会给出播放错误。
