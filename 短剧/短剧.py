#!/usr/bin/env python3
"""本机短剧：写剧本、配音、把图片或视频收成一集。出图和出视频仍用 ComfyUI。"""

import argparse
import json
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
FFMPEG = "/opt/homebrew/bin/ffmpeg"
OLLAMA = "http://127.0.0.1:11434/api/chat"
LLM = "qwen3:8b"
TTS_MODEL = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16"
WIDTH, HEIGHT = 480, 832
FONT = "/System/Library/Fonts/STHeiti Medium.ttc"
FONT_INDEX = 1
SPEAKERS = ("Vivian", "Serena", "Dylan", "Uncle_Fu", "Eric")
FRAMES = {
    "竖屏 9:16": (768, 1344, 480, 832),
    "竖屏 3:4": (768, 1024, 480, 640),
    "方屏 1:1": (1024, 1024, 640, 640),
    "横屏 4:3": (1024, 768, 640, 480),
    "横屏 16:9": (1344, 768, 832, 480),
}
GENRES = ("现实都市", "甜宠", "虐恋", "悬疑", "古装")
PROMPT_EXAMPLE = (
    "近景。林晚：二十五岁东亚女性，鹅蛋脸，黑长直发，细眉，穿米色风衣。"
    "雨夜便利店玻璃门前，她撑着透明伞停下，回头看见程澈。"
    "霓虹映在湿地面，侧光，浅景深，竖屏 9:16，现实都市短剧，电影感实拍。"
)

SPEAKER_HINT = {
    "Vivian": "年轻女声",
    "Serena": "女声",
    "Dylan": "年轻男声",
    "Uncle_Fu": "老年男声",
    "Eric": "旁白",
}


def project_dir(name):
    if not name or name != Path(name).name:
        raise SystemExit("集名不能包含路径")
    path = OUTPUT / name
    path.mkdir(parents=True, exist_ok=True)
    for folder in ("图片", "视频", "音频", "成片"):
        (path / folder).mkdir(exist_ok=True)
    return path


def load_script(path):
    file = path / "剧本.json"
    if not file.exists():
        raise SystemExit(f"还没有剧本：{file}。先运行 写剧本。")
    return json.loads(file.read_text(encoding="utf-8"))


def save_script(path, data):
    (path / "剧本.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def stamp(seconds):
    ms = max(0, int(round(seconds * 1000)))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def shot_duration(shot):
    planned = float(shot.get("seconds") or 4)
    spoken = float(shot.get("audio_seconds") or 0)
    return max(planned, spoken, 2.0)


def write_srt(path, data):
    lines = []
    cursor = 0.0
    index = 1
    for shot in data["shots"]:
        dur = shot_duration(shot)
        text = (shot.get("dialogue") or "").strip()
        if text:
            lines.append(str(index))
            lines.append(f"{stamp(cursor)} --> {stamp(cursor + dur)}")
            lines.append(text)
            lines.append("")
            index += 1
        cursor += dur
    (path / "字幕.srt").write_text("\n".join(lines), encoding="utf-8")


def voice_only(person):
    return "旁白" in (person.get("role") or "")


def write_readable(path, data):
    parts = [f"# {data['title']}", "", data.get("logline", ""), "", "## 角色", ""]
    for person in data["characters"]:
        if voice_only(person):
            parts += [
                f"### {person['name']}（旁白，不入画）",
                f"- 声音：{person['speaker']}，{person.get('voice', '')}",
                "",
            ]
            continue
        parts += [
            f"### {person['name']}（{person.get('role', '')}）",
            f"- 声音：{person['speaker']}，{person.get('voice', '')}",
            f"- 外貌锁定：{person['lock']}",
            "- 定妆：用 ComfyUI「文生图-Z-Image-Turbo」，提示词见 定妆提示词.txt。",
            f"  出图后存成 图片/定妆-{person['name']}.png，后面每镜图生视频都用这张做首帧。",
            "",
        ]
    parts += [
        "## 分镜",
        "",
        "每一镜的文生图提示词已经写进该角色的外貌锁定句。",
        "图生视频打开「图生视频-Wan2.2-5B」，Load Image 用定妆图或本镜图片。",
        "图片存到 图片/01.png 这种文件名，视频存到 视频/01.mp4，再运行「成片」。",
        "",
    ]
    for shot in data["shots"]:
        parts += [
            f"### {shot['id']}  {shot['seconds']}秒",
            f"画面：{shot.get('scene', '')}",
            f"对白（{shot.get('speaker', '')}）：{shot.get('dialogue', '')}",
            "",
            "文生图：",
            shot.get("image_prompt", ""),
            "",
            "图生视频动作：",
            shot.get("video_prompt", ""),
            "",
        ]
    (path / "分镜.md").write_text("\n".join(parts), encoding="utf-8")


def write_looks(path, data):
    frame = data.get("frame") or "竖屏 9:16"
    genre = data.get("genre") or "现实都市"
    lines = []
    for person in data["characters"]:
        if voice_only(person) or person.get("lock") in ("", "无"):
            continue
        prompt = (
            f"{person['lock']}。正面半身定妆照，看向镜头，纯色背景，"
            f"{frame}，{genre}短剧，电影感侧光，真实皮肤纹理，浅景深，摄影级画质"
        )
        lines += [f"# {person['name']}", prompt, ""]
    (path / "定妆提示词.txt").write_text("\n".join(lines), encoding="utf-8")


def pick_speaker(raw, role):
    if raw in SPEAKERS:
        return raw
    text = f"{raw or ''}{role or ''}"
    if "旁白" in text or "叙述" in text:
        return "Eric"
    if "老" in text:
        return "Uncle_Fu"
    if "男" in text:
        return "Dylan"
    if "二" in text or "配" in text:
        return "Serena"
    return "Vivian"


def extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0]
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("模型没有返回 JSON")
    return json.loads(text[start : end + 1])


def chat_json(prompt):
    payload = {
        "model": LLM,
        "stream": False,
        "think": False,
        "format": "json",
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0.7},
    }
    req = urllib.request.Request(
        OLLAMA,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.URLError as exc:
        raise SystemExit(f"Ollama 没连上：{exc.reason}。先确认 ollama 已启动。") from exc
    content = body.get("message", {}).get("content") or ""
    return extract_json(content)


def ask_llm(outline, shots, frame, genre):
    speaker_line = "、".join(f"{name}（{SPEAKER_HINT[name]}）" for name in SPEAKERS)
    prompt = f"""你是短剧编剧，负责把用户的粗略大纲润色成分镜提示词。只输出一个 JSON 对象，不要解释。
保留用户写的人物、地点、关系和情节，不要改成另一个故事。
说话人只能从这些里面选：{speaker_line}。
写 {shots} 个分镜，画幅是{frame}，类型是{genre}。每镜 4 到 6 秒。第 1 镜必须是近景或特写钩子，观众 3 秒内能看懂谁和谁发生了什么。
对白口语、很短，适合配音。旁白用 Eric，对白里不要加角色名。
每个角色的 lock 用一句写实中文锁死年龄、脸型、发型、五官、服装，全剧不要改。

image_prompt 用中文，按这个顺序写：景别，角色名和 lock 原句，动作，场景和时间，光线，镜头，画幅，类型。
video_prompt 按时间顺序写完这一镜的动作。大纲里如果有说话之后的动作，比如转身、离开、伸手，必须写成「先说完，然后转身」。最后加「不要改变人物外貌和服装」。不要只写一个很小的表情。

用户如果只写「雨夜，女主在便利店遇见前男友」，画面提示词要润色成这样：
{PROMPT_EXAMPLE}
对应的 video_prompt：「她先说完这句，然后转身离开，背影走进雨里。不要改变人物外貌和服装。」
不要输出「一个美女在下雨」这种没有人物、地点和光线的句子。

大纲：{outline}

按这个结构输出：
{{
  "title": "剧名",
  "logline": "一句话",
  "characters": [
    {{"name": "林晚", "role": "女主", "speaker": "Vivian", "lock": "二十五岁东亚女性，鹅蛋脸，黑长直发，细眉，穿米色风衣", "voice": "压低声音，语速偏慢"}}
  ],
  "shots": [
    {{"id": "01", "seconds": 5, "characters": ["林晚"], "scene": "雨夜便利店门口", "dialogue": "你怎么在这。", "speaker": "Vivian", "emotion": "意外", "image_prompt": "{PROMPT_EXAMPLE}", "video_prompt": "镜头从伞沿落到她的眼睛，脚步顿住。不要改变人物外貌和服装。"}}
  ]
}}"""
    return chat_json(prompt)


def fit_prompt(text, frame, genre):
    cleaned = str(text or "").strip()
    for word in ("竖屏构图，电影感，真实皮肤，浅景深", "竖屏构图", "横屏构图", "方屏构图"):
        cleaned = cleaned.replace(word, "")
    for name in FRAMES:
        cleaned = cleaned.replace(name, "")
    cleaned = cleaned.strip("，。 ")
    tail = f"{frame}，{genre}短剧，电影感实拍，真实皮肤，浅景深"
    if tail not in cleaned:
        cleaned = f"{cleaned}。{tail}" if cleaned else tail
    return cleaned


def fit_motion(text):
    cleaned = str(text or "").strip().strip("，。 ")
    if "不要改变人物外貌" not in cleaned:
        cleaned = f"{cleaned}。不要改变人物外貌和服装" if cleaned else "不要改变人物外貌和服装"
    if "动作要做完" not in cleaned:
        cleaned += "。按句子顺序把每个动作做完，幅度清楚，不要停在第一帧"
    return cleaned


def polish_shot(outline, data, shot, hint, frame, genre):
    locks = []
    for name in shot.get("characters") or []:
        person = next((item for item in data["characters"] if item["name"] == name), None)
        if person and person.get("lock"):
            locks.append(f"{name}：{person['lock']}")
    prompt = f"""你是短剧提示词润色。只输出 JSON，不要解释。
保留用户指定的人物、地点和动作，只把句子补完整，不要改成另一个故事。
画幅：{frame}
类型：{genre}
全剧大纲：{outline}
这一镜：{shot.get("scene") or ""}
对白：{shot.get("dialogue") or ""}
外貌锁定：{"；".join(locks) or "无"}
用户本镜提示：{hint}

润色事例。用户写「她回头看见他」，应写成：
image_prompt：「{PROMPT_EXAMPLE}」
video_prompt：「她说完这句，然后转身离开，背影走进雨里。不要改变人物外貌和服装。」

输出：{{"image_prompt": "", "video_prompt": ""}}"""
    result = chat_json(prompt)
    return {
        "image_prompt": fit_prompt(result.get("image_prompt") or hint, frame, genre),
        "video_prompt": fit_motion(result.get("video_prompt") or shot.get("video_prompt") or hint),
    }


def normalize(data, shots, frame="竖屏 9:16", genre="现实都市"):
    characters = []
    for person in data.get("characters") or []:
        name = str(person.get("name") or "").strip()
        if not name:
            continue
        role = str(person.get("role") or "").strip()
        lock = str(person.get("lock") or "").strip()
        if "旁白" in role:
            lock = ""
        elif not lock or lock == "无":
            lock = f"{name}，{role or '角色'}，服装和发型全剧保持一致"
        characters.append(
            {
                "name": name,
                "role": role,
                "speaker": pick_speaker(person.get("speaker"), role),
                "lock": lock,
                "voice": str(person.get("voice") or "语速中等，情绪清楚").strip(),
            }
        )
    if not characters:
        raise SystemExit("剧本里没有角色")
    by_name = {person["name"]: person for person in characters}

    cleaned = []
    for index, shot in enumerate((data.get("shots") or [])[:shots], start=1):
        names = shot.get("characters") or []
        if isinstance(names, str):
            names = [part.strip() for part in names.replace("，", ",").split(",") if part.strip()]
        names = [name for name in names if name in by_name]
        if not names:
            names = [characters[0]["name"]]
        speaker = shot.get("speaker")
        if speaker not in SPEAKERS:
            speaker = by_name[names[0]]["speaker"]
        seconds = shot.get("seconds") or 5
        try:
            seconds = int(seconds)
        except (TypeError, ValueError):
            seconds = 5
        seconds = min(8, max(3, seconds))
        image_prompt = str(shot.get("image_prompt") or "").strip()
        scene = str(shot.get("scene") or "").strip()
        missing = [
            name for name in names
            if by_name[name]["lock"] not in ("", "无")
            and by_name[name]["lock"] not in image_prompt
        ]
        if missing:
            prefix = "，".join(f"{name}：{by_name[name]['lock']}" for name in missing)
            image_prompt = "。".join(part for part in (prefix, image_prompt or scene) if part)
        image_prompt = fit_prompt(image_prompt, frame, genre)
        cleaned.append(
            {
                "id": f"{index:02d}",
                "seconds": seconds,
                "characters": names,
                "scene": scene,
                "dialogue": str(shot.get("dialogue") or "").strip(),
                "speaker": speaker,
                "emotion": str(shot.get("emotion") or "").strip(),
                "image_prompt": image_prompt,
                "video_prompt": fit_motion(shot.get("video_prompt") or scene),
            }
        )
    if not cleaned:
        raise SystemExit("剧本里没有分镜")
    return {
        "title": str(data.get("title") or "未命名").strip(),
        "logline": str(data.get("logline") or "").strip(),
        "frame": frame if frame in FRAMES else "竖屏 9:16",
        "genre": genre if genre in GENRES else "现实都市",
        "characters": characters,
        "shots": cleaned,
    }


def write_episode(name, outline, shots, frame="竖屏 9:16", genre="现实都市"):
    print(f"正在写《{name}》，用 {LLM}。这时不要在 ComfyUI 里出图或出视频。")
    data = normalize(ask_llm(outline, shots, frame, genre), shots, frame, genre)
    path = project_dir(name)
    save_script(path, data)
    write_srt(path, data)
    write_readable(path, data)
    write_looks(path, data)
    print(f"剧本已写入 {path}")
    print(f"角色 {len(data['characters'])} 个，分镜 {len(data['shots'])} 个。")
    print("下一步：按 定妆提示词.txt 和 分镜.md 在 ComfyUI 出图、出视频，再运行配音和成片。")


def synthesize(path):
    data = load_script(path)
    import numpy as np
    import soundfile as sf
    from mlx_audio.tts.utils import load

    print("正在加载中文语音。如果 ComfyUI 正在出图或出视频，先等它跑完。")
    model = load(TTS_MODEL)
    for shot in data["shots"]:
        text = shot.get("dialogue") or ""
        wav = path / "音频" / f"{shot['id']}.wav"
        if not text or text in {"无", "无对白", "（无对白）"}:
            wav.unlink(missing_ok=True)
            shot.pop("audio_seconds", None)
            print(f"{shot['id']} 无对白，跳过")
            continue
        person = next(
            (item for item in data["characters"] if item["speaker"] == shot["speaker"]),
            None,
        )
        instruct = shot.get("emotion") or ""
        if person and person.get("voice"):
            instruct = f"{person['voice']}，{instruct}".strip("，")
        print(f"{shot['id']} {shot['speaker']}：{text}")
        chunks = []
        rate = 24000
        for result in model.generate_custom_voice(
            text=text,
            speaker=shot["speaker"],
            language="Chinese",
            instruct=instruct or None,
        ):
            chunks.append(np.asarray(result.audio).reshape(-1))
            rate = result.sample_rate
        audio = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
        sf.write(wav, audio, rate)
        shot["audio_seconds"] = round(len(audio) / rate + 0.3, 2)
    save_script(path, data)
    write_srt(path, data)
    print(f"配音完成：{path / '音频'}")


def ffmpeg(args):
    result = subprocess.run(
        [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", *args],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise SystemExit(detail or "ffmpeg 失败")


def find_media(folder, stem, suffixes):
    for suffix in suffixes:
        candidate = folder / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def subtitle_image(text, dest):
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    if not text:
        image.save(dest)
        return
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT, 34, index=FONT_INDEX)
    max_width = WIDTH - 48
    lines = []
    current = ""
    for char in text:
        trial = current + char
        if draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = char
    if current:
        lines.append(current)
    lines = lines[:3]
    line_height = 46
    block = line_height * len(lines)
    top = HEIGHT - 72 - block
    draw.rectangle((16, top - 12, WIDTH - 16, top + block + 8), fill=(0, 0, 0, 150))
    for index, line in enumerate(lines):
        width = draw.textlength(line, font=font)
        x = (WIDTH - width) / 2
        y = top + index * line_height
        draw.text((x, y), line, font=font, fill=(255, 255, 255, 255))
    image.save(dest)


def silence(dest, seconds):
    import numpy as np
    import soundfile as sf

    count = int(max(seconds, 0.2) * 44100)
    sf.write(dest, np.zeros(count, dtype=np.float32), 44100)


def join_shots(path):
    videos = sorted((path / "视频").glob("*.mp4"))
    dest = path / "成片.mp4"
    if not videos:
        raise SystemExit(f"还没有分镜视频：{path / '视频'}")
    if len(videos) == 1:
        shutil.copyfile(videos[0], dest)
        return dest
    work = path / "成片"
    work.mkdir(exist_ok=True)
    list_file = work / "join.txt"
    list_file.write_text(
        "".join(f"file '{video.resolve()}'\n" for video in videos),
        encoding="utf-8",
    )
    try:
        ffmpeg(["-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(dest)])
    except SystemExit:
        probe = subprocess.run(
            [
                "/opt/homebrew/bin/ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0",
                str(videos[0]),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        width, height = probe.stdout.strip().split(",")
        ffmpeg([
            "-f", "concat", "-safe", "0", "-i", str(list_file),
            "-vf",
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            str(dest),
        ])
    return dest


def assemble(path):
    data = load_script(path)
    work = path / "成片"
    list_file = work / "list.txt"
    entries = []
    for shot in data["shots"]:
        duration = shot_duration(shot)
        clip = work / f"{shot['id']}.mp4"
        audio = path / "音频" / f"{shot['id']}.wav"
        if not audio.exists():
            audio = work / f"{shot['id']}-silence.wav"
            silence(audio, duration)
        sub = work / f"{shot['id']}-sub.png"
        subtitle_image(shot.get("dialogue") or "", sub)
        video = find_media(path / "视频", shot["id"], (".mp4", ".mov", ".webm"))
        image = find_media(path / "图片", shot["id"], (".png", ".jpg", ".jpeg", ".webp"))
        scale = (
            f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,"
            f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2:black,fps=24,"
            f"trim=duration={duration:.3f},setpts=PTS-STARTPTS"
        )
        audio_filter = (
            f"aformat=sample_rates=44100:channel_layouts=stereo,"
            f"apad,atrim=duration={duration:.3f},asetpts=PTS-STARTPTS"
        )
        if video:
            video_filter = (
                f"{scale.rsplit(',trim', 1)[0]},"
                f"tpad=stop_mode=clone:stop_duration={duration:.3f},trim=duration={duration:.3f},setpts=PTS-STARTPTS"
            )
            ffmpeg(
                [
                    "-i", str(video),
                    "-i", str(audio),
                    "-i", str(sub),
                    "-filter_complex",
                    f"[0:v]{video_filter}[bg];[bg][2:v]overlay=0:0:format=auto,format=yuv420p[v];[1:a]{audio_filter}[a]",
                    "-map", "[v]", "-map", "[a]",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-t", f"{duration:.3f}",
                    str(clip),
                ]
            )
        elif image:
            ffmpeg(
                [
                    "-loop", "1", "-framerate", "24", "-i", str(image),
                    "-i", str(audio),
                    "-i", str(sub),
                    "-filter_complex",
                    f"[0:v]{scale}[bg];[bg][2:v]overlay=0:0:format=auto,format=yuv420p[v];[1:a]{audio_filter}[a]",
                    "-map", "[v]", "-map", "[a]",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-t", f"{duration:.3f}",
                    str(clip),
                ]
            )
        else:
            ffmpeg(
                [
                    "-f", "lavfi", "-i", f"color=c=black:s={WIDTH}x{HEIGHT}:r=24:d={duration:.3f}",
                    "-i", str(audio),
                    "-i", str(sub),
                    "-filter_complex",
                    f"[0:v]fps=24,trim=duration={duration:.3f},setpts=PTS-STARTPTS[bg];"
                    f"[bg][2:v]overlay=0:0:format=auto,format=yuv420p[v];[1:a]{audio_filter}[a]",
                    "-map", "[v]", "-map", "[a]",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-t", f"{duration:.3f}",
                    str(clip),
                ]
            )
        entries.append(f"file '{clip.name}'")
        print(f"{shot['id']} 已收入成片")
    list_file.write_text("\n".join(entries) + "\n", encoding="utf-8")
    final = path / "成片.mp4"
    ffmpeg(["-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(final)])
    write_srt(path, data)
    print(f"成片：{final}")
    print(f"字幕：{path / '字幕.srt'}")


def speak_to_file(text, speaker, instruct, dest):
    import numpy as np
    import soundfile as sf
    from mlx_audio.tts.utils import load

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if speaker not in SPEAKERS:
        speaker = "Vivian"
    if not text or text in {"无", "无对白", "（无对白）"}:
        sf.write(dest, np.zeros(int(0.4 * 24000), dtype=np.float32), 24000)
        return
    model = load(TTS_MODEL)
    chunks = []
    rate = 24000
    for result in model.generate_custom_voice(
        text=text,
        speaker=speaker,
        language="Chinese",
        instruct=instruct or None,
    ):
        chunks.append(np.asarray(result.audio).reshape(-1))
        rate = result.sample_rate
    audio = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
    sf.write(dest, np.clip(audio, -1, 1), rate, subtype="PCM_16")


def main():
    parser = argparse.ArgumentParser(prog="短剧", description="本机短剧：写剧本、配音、成片")
    sub = parser.add_subparsers(dest="cmd", required=True)

    write = sub.add_parser("写剧本")
    write.add_argument("--集", required=True)
    write.add_argument("--大纲", required=True)
    write.add_argument("--镜数", type=int, default=6)

    voice = sub.add_parser("配音")
    voice.add_argument("--集", required=True)

    cut = sub.add_parser("成片")
    cut.add_argument("--集", required=True)

    say = sub.add_parser("说")
    say.add_argument("--文本", required=True)
    say.add_argument("--说话人", default="Vivian")
    say.add_argument("--情绪", default="")
    say.add_argument("--输出", required=True)

    args = parser.parse_args()
    if args.cmd == "说":
        speak_to_file(args.文本, args.说话人, args.情绪, args.输出)
        return
    if args.cmd == "写剧本":
        count = min(12, max(2, args.镜数))
        write_episode(args.集, args.大纲, count)
    elif args.cmd == "配音":
        synthesize(project_dir(args.集))
    else:
        assemble(project_dir(args.集))


if __name__ == "__main__":
    main()
