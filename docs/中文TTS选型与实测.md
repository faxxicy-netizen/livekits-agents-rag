# 中文 TTS 选型与本机实测

目标机器：Windows，i5-13490F、RTX 3060 Ti 8GB、32GB 内存。目标是实时对话中的中文清晰度、自然语速、低首段等待与稳定打断；不需要声音克隆。本轮不做前端多尺寸适配。

## 官方项目对比

| 项目 | 适合之处 | 这次作业的取舍 |
| --- | --- | --- |
| [Kokoro-82M-v1.1-zh](https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh) | 82M 参数，中文专用训练，100 个中文说话人，Apache 2.0；非扩散解码，支持语速参数 | 先接入并在本机实际测量。无需参考音频，无需 Linux、Docker 或编译 CUDA 扩展。适合固定客服音色与短句合成。 |
| [CosyVoice 3](https://github.com/QwenAudio/CosyVoice) | 0.5B 模型，中文及方言、韵律控制、文本和音频双流式输出 | 若试听后仍希望更有表现力，优先进一步评估。依赖、文本前端和加速运行时更复杂；官方最低 150ms 是其环境的结果，不能视为本机承诺。 |
| [Qwen3-TTS 0.6B CustomVoice](https://github.com/QwenLM/Qwen3-TTS) | 预设中文声音，无需自行准备克隆参考音频；官方列出流式能力 | 值得作为另一候选。模型规模和自回归生成成本更高，Windows 下建议的 FlashAttention 加速安装也更复杂。当前未在本机安装或测量。 |
| [F5-TTS](https://github.com/SWivid/F5-TTS) | 中文声音克隆，支持分块推理和语速控制 | 需要参考音频，流匹配推理存在步数与速度取舍。官方 L20/TensorRT 数据不能直接外推到 3060 Ti。当前作业无声音克隆需求，先不引入。 |
| [IndexTTS 2.5](https://github.com/index-tts/index-tts) | 中文、情绪、声音克隆和语速控制 | 更偏向表现力与配音控制；官方提醒 Windows 下 DeepSpeed 安装较难。当前不作为低依赖实时对话的首选。 |

Kokoro 已接入本机，其他候选未安装或测量。用户要求快速替换并自行用浏览器验证，因此本轮停止综合测速与端到端验证，没有给出未经测量的显存需求或首包数字。参数量也不等于总显存：还需计算声码器、中间张量、缓存、驱动与桌面占用。

## 已接入的实现

- 默认 `Kokoro-82M-v1.1-zh`，使用 CUDA 版 PyTorch 2.8，FP32 推理；不用扩散采样或 FlashAttention。
- 模型和三个音色从上游固定 revision 下载，核对权重 SHA256。运行时使用本地路径，不临时访问 Hugging Face。
- LiveKit 官方 `setup_fnc` 预热模型；Windows 的作业线程共享缓存和推理锁，避免每次通话重复装载 GPU 模型。
- 按中文句号、问号和较长句子的逗号拆段。每段完成即送出 PCM，后续句子继续合成；这是分句输出，不宣称模型内部逐音频 token 流式生成。
- 默认速度 `1.10`，可以用 `.env` 的 `LOCAL_TTS_SPEED` 调节；这是模型的时长控制参数，不是简单提高播放采样率。
- 原始回答与引用保持原文，朗读层不做型号或领域术语替换。英文部分交给模型的本地 eSpeak 前端转为音素。
- 打断取消后，不再继续合成后续分句；正在执行的单个 GPU 推理会完成，结果不会继续播放。

配置例子：

```dotenv
TTS_PROVIDER=local
LOCAL_TTS_BACKEND=kokoro
LOCAL_TTS_DEVICE=cuda
LOCAL_TTS_VOICE=zf_001
LOCAL_TTS_SPEED=1.10
```

已下载音色：`zf_001`、`zf_017` 为女声，`zm_010` 为男声。修改后重启应用。不需要购买语音 API Key。CPU 备用模式可设 `LOCAL_TTS_DEVICE=cpu`，需自行实测速度。

旧 VITS 仍可复现比较：`LOCAL_TTS_BACKEND=vits`、`LOCAL_TTS_DIR=runtime/models/vits-zh`。它不再作为默认音色。

## 测量方法

```powershell
uv run python scripts/benchmark_tts.py
uv run python scripts/check_speech.py
uv run python scripts/smoke_voice.py
```

合成对比使用同一段中文，先预热，然后重复三次；分别记录首段耗时、整段合成耗时、音频长度和 RTF。RTF = 合成秒数 / 音频秒数，越小越快。首段耗时只包含 TTS，不包含用户停顿检测、STT、DeepSeek、WebRTC 缓冲。

显存数据使用 PyTorch 的峰值 allocated/reserved 计数，不能当作整个系统显存。真实对话延迟另由 WebRTC 测试记录。音质最终以用户试听为准，自动识别回测只检查可懂度，不能代表主观自然度。

上述命令为可选检查，不是本轮已完成的验收。测速脚本执行成功后会将结果写入 `runtime/tts-comparison/benchmark.json` 与同目录 WAV 文件。本轮交付直接在浏览器试听，由用户判断音色和节奏。
