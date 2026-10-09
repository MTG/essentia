# 第三方许可说明

QuietTune 使用真实开源音频组件和官方预训练模型。发行或修改这些组件时应保留各自版权与许可文件。

QuietTune 新增应用代码采用 GNU AGPL v3，许可原文见应用根目录 `LICENSE`；它与 fork 根目录的上游 `COPYING.txt` 一致。桌面构建将该原文保存为 `resources/QuietTune-LICENSE.txt`。依赖、模型权重、音乐和 Microsoft 可再发行组件分别适用各自许可，不因本项目代码开源而改为 AGPL。

| 资源 | 当前使用内容 | 官方许可来源 | 使用义务 |
| --- | --- | --- | --- |
| Electron 44.7.0、Node.js 22.18.0 | 桌面窗口与 Windows WASM 子进程 | https://github.com/electron/electron/blob/main/LICENSE 、https://github.com/nodejs/node/blob/v22.18.0/LICENSE | MIT 及所含第三方声明；Electron 自带 LICENSE/第三方声明，Node 原文随运行环境分发。 |
| Python 3.11.6 | 官方 Windows x64 嵌入式发行版 | https://www.python.org/ftp/python/3.11.6/ | PSF 许可；保留发行包的 LICENSE.txt。 |
| Microsoft Visual C++ 运行组件 | 私有 Python 旁的 msvcp140、vcruntime140、concrt140 | https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist | Microsoft 可再发行组件，依其分发条款使用，不能将其声明为本项目开源组件。Windows 11 自带的系统基础库不重复打包。 |
| Essentia | 官方音乐分析算法，Linux Python wheel | https://github.com/MTG/essentia/blob/master/COPYING.txt | AGPL-3.0；分发及向网络用户提供修改版时，遵循对应源码提供义务。商业许可需向权利方申请。 |
| essentia.js 0.1.3 | Windows WebAssembly TensorflowInputMusiCNN | https://github.com/MTG/essentia.js | AGPL-3.0；保留许可并履行源码提供义务。 |
| MSD MusiCNN 权重 | msd-musicnn-1.pb，元数据版本 1 | https://essentia.upf.edu/models/feature-extractors/musicnn/ | 官方非商业模型，SA/ND 声明存在差异，详见下文；权重不随本仓库分发。 |
| 激烈分类器 | mood_aggressive-msd-musicnn-1.pb，元数据版本 2 | https://essentia.upf.edu/models/classification-heads/mood_aggressive/ | 官方非商业模型，SA/ND 声明存在差异。文件名后缀 1 不代表元数据版本 1。 |
| 放松分类器 | mood_relaxed-msd-musicnn-1.pb，元数据版本 2 | https://essentia.upf.edu/models/classification-heads/mood_relaxed/ | 官方非商业模型，SA/ND 声明存在差异，不将其他模型视为同一许可。 |
| TensorFlow | GraphDef 推理 | https://github.com/tensorflow/tensorflow/blob/master/LICENSE | Apache-2.0。 |
| librosa | 辅助频谱与节奏特征 | https://github.com/librosa/librosa/blob/main/LICENSE.md | ISC。 |
| FFmpeg | 解码、元数据、EBU R128、真峰值 | https://ffmpeg.org/legal.html | 依构建选项为 LGPL/GPL；本机使用 Gyan 构建，实际分发需核查其构建许可。Docker 使用 Debian 软件包，不将其许可视为模型许可。 |
| React、Vite、Lucide、Recharts、WaveSurfer.js | 前端组件 | 各 npm 包随附 LICENSE | 依各包许可保留声明；其中 React、Vite、Lucide、Recharts、WaveSurfer.js 为 MIT/ISC 类宽松许可，具体以锁定包附带文件为准。 |
| Vibe Ace | 本地验证使用的真实音乐 | https://librosa.org/data/audio/Kevin_MacLeod_-_Vibe_Ace.txt | Kevin MacLeod — Vibe Ace，CC BY 3.0，来源 Free Music Archive。许可文本保存于 .Codex/fixtures/Vibe-Ace-license.txt，音乐不纳入源代码分发。 |

模型共同许可声明来源：https://essentia.upf.edu/models.html 和 https://essentia.upf.edu/models/LICENSE 。下载器仅缓存所选三个模型，不把同一许可强加给其他模型。

截至 2026-10-09，模型列表页写明 CC BY-NC-SA 4.0，许可说明页 https://essentia.upf.edu/licensing_information.html 写明 CC BY-NC-ND 4.0；模型 `LICENSE` 文件标题和正文为 ND，但包含 SA legalcode 链接。本项目记录该冲突，不替权利方确定允许再分发或修改的版本。模型需由使用者从官方地址下载，用途和再分发条件以权利方确认的许可为准。本次 GitHub 发布仅包含代码和构建配置，不包含 `.pb`、模型元数据副本或模型缓存清单。

模型原始 JSON 保存了作者、训练数据、框架版本、类别顺序及张量协议；`models/manifest.json` 保存逐文件来源与本地 SHA-256。官方这些 JSON 未提供权重摘要，因此本地摘要只能验证缓存未变化，不能宣称是官方发布的校验值。

本机测试音乐与模型权重均排除在版本控制之外。桌面安装包包含三个模型，用于本地非商业使用，音乐不随安装包分发。模型原始署名与许可元数据随包保留。

已检查实际打包的 Gyan FFmpeg 9.0.1：构建参数包含 `--enable-gpl --enable-version3`，`ffmpeg -L` 明确声明 GPL 第 3 版或更高版本，因此本桌面包的 FFmpeg 按 GPLv3+ 处理。保留原始 LICENSE、README 及 https://www.gyan.dev/ffmpeg/builds/ 的构建与源码信息。向他人分发时必须提供与所分发 FFmpeg/Essentia 二进制匹配的对应源码及许可证要求的构建资料，只有链接或本声明不能替代全部源码义务。

桌面运行环境的 `licenses/` 收集 QuietTune、Node、FFmpeg 原文、前端依赖版权文件和 Python 依赖锁；Python 包的原始许可与 dist-info 一起保留。GitHub 中的公开源码不包含这些二进制运行环境或本机生成的桌面安装包。
