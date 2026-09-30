# comfyui-duanju

本机简体中文短剧流水线。在 Mac 上用 ComfyUI 从大纲写到定妆、五秒镜头、配音、口型和字幕，再把多镜接成一集。不调用即梦、可灵、Vidu 或其他云端视频接口。

这套流程是按 Apple 芯片、统一内存、MPS 调的。没有 NVIDIA，也不能用 fp8。32GB 内存一次只能载入一个大模型，所以剧本、出图、出视频、配音会轮流占用内存，不要同时开第二个任务。

## 一次运行做什么

在 ComfyUI 左侧「应用」里打开「短剧」。点一次运行，完成**一镜**，视频固定 **5 秒、24 帧/秒、120 帧**。

顺序是：

1. 用本机 Ollama 的 `qwen3:8b` 把大纲润色成分镜。已有剧本且没有勾选「重写」时，直接读现成剧本。
2. 用 Z-Image Turbo 出一张定妆静帧。
3. 卸掉出图模型，再载入 Wan2.2 TI2V 5B，从这张静帧接着做图生视频。
4. 用 Qwen3-TTS 配这一句对白。
5. 用 Wav2Lip 只改嘴巴。侧脸或检不到脸时保持原画面。
6. 用黑体把对白烧进画面，并写出 `字幕.srt`。
7. 把这一镜存进该集目录，再和已有分镜接成 `成片.mp4`。

五秒不是一次采样拉满。这台机器上一次生成太长的视频，后半段会散成色块，强行把短片拉长又会卡顿。所以每段只生成 17 帧，再用最后一帧接着下一段，直到 120 帧。

时间是这样分的：

- 前两秒面对镜头说话，口型和这句对白对齐。
- 大纲或本镜提示里有「转身」或「离开」时，后三秒再做转身，直到背对镜头。
- 对白比五秒短时，说完后面继续画面，不再把前面的帧重复来凑时长。

改镜号再运行一次，新的一镜会接到同一集后面。

## 目录

```text
custom_nodes/comfyui_duanju/   ComfyUI 节点：剧本、模型切换、五秒视频、配音、字幕、归档
custom_nodes/Halo-Lipsy/       本机口型。已改成 OpenCV YuNet 找脸，正脸才贴嘴
workflows/短剧.app.json        ComfyUI 应用模式工作流
短剧/短剧.py                   命令行：写剧本、配音、拼接、单句试听
短剧/使用.txt                  给操作时看的短说明
短剧/output/                   每一集的剧本、分镜、字幕、定妆图和单镜视频
```

仓库里没有成片，也没有 `__pycache__`、`.venv` 和模型权重。

节点里写死了本机路径。换机器时要改 `custom_nodes/comfyui_duanju/__init__.py` 开头这三项：

- `DUANJU`：`短剧.py` 的绝对路径
- `DUANJU_PYTHON`：短剧虚拟环境里的 Python
- `OLLAMA`：`ollama` 可执行文件，Homebrew 一般是 `/opt/homebrew/bin/ollama`

`短剧.py` 里的 ffmpeg 路径是 `/opt/homebrew/bin/ffmpeg`。

## 环境

在这台机器上验证过的组合：

- macOS，Apple MPS，约 32GB 统一内存
- ComfyUI 0.28，前端应用模式能打开后缀为 `app.json` 的工作流
- Python 3.12
- PyTorch 带 MPS
- Homebrew 的 ffmpeg。这个 ffmpeg 没有 libass，字幕是用 Pillow 画上去的
- Ollama，模型 `qwen3:8b`
- 另一套虚拟环境里的 mlx-audio，用来跑 Qwen3-TTS

ComfyUI 要用 PyTorch 的注意力，否则 MPS 上的视频后半段更容易散：

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
python main.py --listen 127.0.0.1 --port 8188 --use-pytorch-cross-attention
```

打开 http://127.0.0.1:8188 。

## 安装

1. 准备一份能在本机跑起来的 ComfyUI。不要再克隆第二份 Desktop 安装。
2. 把本仓库的 `custom_nodes/comfyui_duanju` 和 `custom_nodes/Halo-Lipsy` 放进那份 ComfyUI 的 `custom_nodes/`。
3. 把 `workflows/短剧.app.json` 放到 ComfyUI 的 `user/default/workflows/`。
4. 把 `短剧/` 放到节点里 `DUANJU` 指向的位置。当前代码写的是 `/Users/alang/devolop/comfyui/短剧/短剧.py`。
5. 给短剧单独建虚拟环境，安装 mlx-audio，不要和 ComfyUI 的 `.venv` 混用：

```bash
cd 短剧
python3.12 -m venv .venv
.venv/bin/pip install mlx-audio soundfile
```

6. 安装并拉取剧本模型：

```bash
ollama pull qwen3:8b
```

7. 把下面的模型放进 ComfyUI 对应目录。全部用 fp16 或 bf16，不要下 fp8。fp8 在 MPS 上会直接失败。Wan 14B 在 32GB 上放不下。

| 文件 | 目录 | 用途 |
| --- | --- | --- |
| `z_image_turbo_bf16.safetensors` | `models/diffusion_models` | 定妆静帧 |
| `qwen_3_4b.safetensors` | `models/text_encoders` | Z-Image 文本编码器，类型用 `lumina2` |
| `ae.safetensors` | `models/vae` | Z-Image 的 VAE |
| `wan2.2_ti2v_5B_fp16.safetensors` | `models/diffusion_models` | 图生视频 |
| `umt5_xxl_fp16.safetensors` | `models/text_encoders` | Wan 文本编码器，类型用 `wan` |
| `wan2.2_vae.safetensors` | `models/vae` | Wan 2.2 的 VAE |
| `wav2lip_gan.pth` | `models/wav2lip` | 口型 |

口型找脸用的是仓库里的 `custom_nodes/Halo-Lipsy/face_detection_yunet_2023mar.onnx`。这台机器上的 OpenCV 5 没有 `CascadeClassifier`，所以不用 Haar。

TTS 权重由 mlx-audio 第一次调用时下载：`mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16`。

## 在应用里做一集

重启 ComfyUI，等自定义节点加载完，再打开「应用」里的「短剧」。页面如果是节点加载前就开着的，先关掉重新打开，否则表单会缺新字段。

表单字段：

| 字段 | 作用 |
| --- | --- |
| 集 | 输出目录名，例如 `第一集` |
| 大纲 | 这一集的故事。会润色成分镜，不会改成另一个故事 |
| 镜号 | 1 到 12。改号再运行，就会接上下一镜 |
| 重写 | 勾上会重新向模型要整集剧本并覆盖 `剧本.json` |
| 画幅 | 竖屏 9:16、竖屏 3:4、方屏 1:1、横屏 4:3、横屏 16:9。同一集用同一个 |
| 类型 | 现实都市、甜宠、虐恋、悬疑、古装 |
| 本镜提示 | 可留空。写了就只润色这一镜，例如「说完这句，转身离开」 |
| 音色 | 默认跟随角色。也可以指定年轻女声、女声、年轻男声、老年男声、旁白 |
| 种子 | 定妆图的采样种子 |
| 主角参考图 1–3 | 可留空。按剧本里的角色顺序贴到定妆图上 |
| 背景参考图 1–2 | 可留空。按镜号选用，只放一张时每镜都用它 |
| 视频参考 1–2 | 可留空。放了就用参考视频开头的画面，不再用静帧拼接 |

参考图留空时，起始画面就是刚生成的定妆图。

提示词按短剧来写，不要只写「一个美女在下雨」。画面提示词的顺序是：景别、外貌锁定、动作、场景、光线、镜头、画幅、类型。例如：

```text
近景。林晚：二十五岁东亚女性，鹅蛋脸，黑长直发，细眉，穿米色风衣。
雨夜便利店玻璃门前，她撑着透明伞停下，回头看见程澈。
霓虹映在湿地面，侧光，浅景深，竖屏 9:16，现实都市短剧，电影感实拍。
```

大纲里如果写了说话之后的动作，例如转身、离开、伸手，动作提示会按「先说完，然后做这个动作」写进去。

音色和角色的对应：

| 选项 | 说话人 | 适合 |
| --- | --- | --- |
| 年轻女声 Vivian | Vivian | 年轻女声 |
| 女声 Serena | Serena | 女声 |
| 年轻男声 Dylan | Dylan | 年轻男声 |
| 老年男声 Uncle_Fu | Uncle_Fu | 老年男声 |
| 旁白 Eric | Eric | 旁白 |

选音色只改配音，不改这一镜用哪张人物参考图。

一镜在这台机器上大约要几十分钟，主要花在五秒视频的八段采样上。内存紧的时候不要再开 Ollama 聊天或其他 ComfyUI 任务。剧本模型用完会执行 `ollama stop qwen3:8b`。

## 输出

每一集写到 `短剧/output/集名/`：

```text
剧本.json       角色锁定、分镜、对白、说话人
分镜.md         给人看的分镜
定妆提示词.txt  静帧提示词
字幕.srt        字幕时间轴
图片/01.png     这一镜的定妆图
视频/01.mp4     这一镜的五秒视频
音频/           命令行配音时的 wav
成片.mp4        已有分镜接在一起的成片，不放进 git
```

`成片.mp4` 和 `__pycache__`、`.venv` 被 `.gitignore` 排除。单镜视频、定妆图、剧本和字幕可以提交。

## 命令行

命令行不负责出图和出视频，那两步在 ComfyUI 里做。在 `短剧/` 目录：

```bash
.venv/bin/python 短剧.py 写剧本 --集 第一集 --大纲 "雨夜，女主在便利店遇见前男友" --镜数 6
.venv/bin/python 短剧.py 配音 --集 第一集
.venv/bin/python 短剧.py 成片 --集 第一集
.venv/bin/python 短剧.py 说 --文本 "你怎么在这。" --说话人 Vivian --情绪 "压低声音" --输出 /tmp/line.wav
```

`写剧本` 的镜数会限制在 2 到 12。`配音` 按剧本里的说话人把每句对白写成 wav。`成片` 把该集 `视频/` 里已有的 mp4 按镜号接起来；编码不一致时会重新压成第一段的尺寸。

双击 `短剧.command` 等于不带参数运行，会打印帮助。

## 画幅

静帧和视频用同一套像素预算，边长都能被 32 整除。

| 画幅 | 静帧 | 视频 |
| --- | --- | --- |
| 竖屏 9:16 | 768×1344 | 480×832 |
| 竖屏 3:4 | 768×1024 | 480×640 |
| 方屏 1:1 | 1024×1024 | 640×640 |
| 横屏 4:3 | 1024×768 | 640×480 |
| 横屏 16:9 | 1344×768 | 832×480 |

视频比静帧小，是为了让 5B 模型能在 32GB 上跑完。静帧清楚、视频软一档是预期情况。如果后半段变成色块，那是采样散了，不是播放器把 480 宽放大造成的。

## 限制

- 一镜五秒。再拉长，这台机器上的 Wan 2.2 5B 后半段会散，解码也会把内存吃满。
- 同一张脸靠每镜重复的外貌锁定句，以及用定妆图当视频第一帧。没有 IP-Adapter、PuLID 或 InstantID 权重，换构图时脸仍可能变。
- 口型只处理正脸。转成侧面或背面后不再贴嘴，避免把嘴贴到后脑勺上。
- 口型是 96×96 的嘴部补丁，近景能对上，仍然能看出是后贴的嘴。
- 采样用 20 步、CFG 4、`euler`、`simple`。不要改回 49 帧、CFG 5、`uni_pc`，那组参数在这台 Mac 上会从大约 0.2 秒开始糊掉。
- 文本编码器必须是 `umt5_xxl_fp16.safetensors`。官方模板默认的 fp8 编码器不能用。
- Z-Image 和 Wan 不能同时留在内存里。节点会在切换前卸载上一组模型。
