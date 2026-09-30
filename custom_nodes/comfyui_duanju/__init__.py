"""把本机短剧的剧本、配音和模型切换接进 ComfyUI。出图和出视频仍用官方节点。"""

import importlib.util
import os
import subprocess
import wave
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import comfy.model_management as model_management
import folder_paths
from comfy_extras.nodes_wan import Wan22ImageToVideoLatent
from nodes import CLIPLoader, CLIPTextEncode, UNETLoader, VAEDecode, VAELoader, common_ksampler

DUANJU = Path("/Users/alang/devolop/comfyui/短剧/短剧.py")
DUANJU_PYTHON = Path("/Users/alang/devolop/comfyui/短剧/.venv/bin/python")
OLLAMA = "/opt/homebrew/bin/ollama"
FONT = "/System/Library/Fonts/STHeiti Medium.ttc"
FONT_INDEX = 1


def _duanju():
    spec = importlib.util.spec_from_file_location("duanju_local", DUANJU)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stop_llm():
    subprocess.run([OLLAMA, "stop", "qwen3:8b"], check=False, capture_output=True)


def _release():
    model_management.unload_all_models()
    model_management.soft_empty_cache()


class DuanjuShot:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "集": ("STRING", {"default": "第一集"}),
                "大纲": ("STRING", {"multiline": True, "default": "雨夜，女主在便利店遇见前男友，说了两句就转身离开"}),
                "镜号": ("INT", {"default": 1, "min": 1, "max": 12}),
                "重写": ("BOOLEAN", {"default": False}),
                "画幅": (["竖屏 9:16", "竖屏 3:4", "方屏 1:1", "横屏 4:3", "横屏 16:9"], {"default": "竖屏 9:16"}),
                "类型": (["现实都市", "甜宠", "虐恋", "悬疑", "古装"], {"default": "现实都市"}),
                "本镜提示": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "可留空。例如：说完这句，转身离开",
                }),
                "音色": ([
                    "跟随角色",
                    "年轻女声 Vivian",
                    "女声 Serena",
                    "年轻男声 Dylan",
                    "老年男声 Uncle_Fu",
                    "旁白 Eric",
                ], {"default": "跟随角色"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("画面提示词", "动作提示词", "对白", "说话人", "情绪", "集", "镜号", "角色序号", "画面宽", "画面高", "视频宽", "视频高")
    FUNCTION = "pick"
    CATEGORY = "短剧"

    def pick(self, 集, 大纲, 镜号, 重写, 画幅, 类型, 本镜提示, 音色="跟随角色"):
        duanju = _duanju()
        frame = 画幅 if 画幅 in duanju.FRAMES else "竖屏 9:16"
        genre = 类型 if 类型 in duanju.GENRES else "现实都市"
        image_w, image_h, video_w, video_h = duanju.FRAMES[frame]
        path = duanju.project_dir(集)
        script_file = path / "剧本.json"
        if 重写 or not script_file.exists():
            print(f"[短剧] 正在按「{类型} / {画幅}」润色《{集}》的大纲。这时不要再开第二个出图任务。")
            data = duanju.normalize(duanju.ask_llm(大纲, max(镜号, 6), frame, genre), 12, frame, genre)
            duanju.save_script(path, data)
            duanju.write_srt(path, data)
            duanju.write_readable(path, data)
            duanju.write_looks(path, data)
        else:
            data = duanju.load_script(path)
        index = min(max(int(镜号), 1), len(data["shots"])) - 1
        shot = data["shots"][index]
        hint = (本镜提示 or "").strip()
        if hint:
            print(f"[短剧] 按本镜提示润色第 {shot['id']} 镜。")
            polished = duanju.polish_shot(大纲, data, shot, hint, frame, genre)
            shot["image_prompt"] = polished["image_prompt"]
            shot["video_prompt"] = polished["video_prompt"]
            duanju.save_script(path, data)
        else:
            shot["image_prompt"] = duanju.fit_prompt(shot.get("image_prompt"), frame, genre)
            shot["video_prompt"] = duanju.fit_motion(shot.get("video_prompt"))
        lead = (data.get("characters") or [{}])[0].get("name")
        names = shot.get("characters") or []
        if lead and lead in names and "转身" in (大纲 or "") and "转身" not in shot["video_prompt"]:
            shot["video_prompt"] = f"说完这句后转身离开，转身要完整能看出来。{shot['video_prompt']}"
        _stop_llm()
        emotion = shot.get("emotion") or ""
        person = next((item for item in data["characters"] if item["speaker"] == shot["speaker"]), None)
        if person and person.get("voice"):
            emotion = f"{person['voice']}，{emotion}".strip("，")
        print(f"[短剧] 第 {shot['id']} 镜，{shot['speaker']}：{shot.get('dialogue', '')}")
        speaker = shot.get("speaker") or ""
        if "旁白" in speaker:
            role_index = -1
        else:
            role_index = next(
                (i for i, item in enumerate(data["characters"]) if item.get("speaker") == speaker),
                0,
            )
        chosen = {
            "年轻女声 Vivian": "Vivian",
            "女声 Serena": "Serena",
            "年轻男声 Dylan": "Dylan",
            "老年男声 Uncle_Fu": "Uncle_Fu",
            "旁白 Eric": "Eric",
        }.get(音色)
        if chosen:
            speaker = chosen
            print(f"[短剧] 音色改为 {speaker}")
        return (
            shot.get("image_prompt") or shot.get("scene") or 大纲,
            shot.get("video_prompt") or shot.get("scene") or "",
            shot.get("dialogue") or "",
            speaker or "Vivian",
            emotion,
            集,
            shot["id"],
            role_index,
            image_w,
            image_h,
            video_w,
            video_h,
        )


class DuanjuZImageStack:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"prompt": ("STRING", {"forceInput": True})}}

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load"
    CATEGORY = "短剧"

    def load(self, prompt):
        _release()
        model = UNETLoader().load_unet("z_image_turbo_bf16.safetensors", "default")[0]
        clip = CLIPLoader().load_clip("qwen_3_4b.safetensors", "lumina2")[0]
        vae = VAELoader().load_vae("ae.safetensors")[0]
        return (model, clip, vae)


class DuanjuWanStack:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load"
    CATEGORY = "短剧"

    def load(self, image):
        _release()
        model = UNETLoader().load_unet("wan2.2_ti2v_5B_fp16.safetensors", "default")[0]
        clip = CLIPLoader().load_clip("umt5_xxl_fp16.safetensors", "wan")[0]
        vae = VAELoader().load_vae("wan2.2_vae.safetensors")[0]
        return (model, clip, vae)


class DuanjuSeconds:
    """五秒成片。每次只生成 17 帧，再从最后一帧接下去，避免把短片拉长造成卡顿。"""

    FPS = 24
    SECONDS = 5
    CHUNK = 17
    NEGATIVE = (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
        "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
        "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
        "杂乱的背景，三条腿，背景人很多，倒着走"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "画面": ("IMAGE",),
                "动作": ("STRING", {"forceInput": True}),
                "宽": ("INT", {"forceInput": True}),
                "高": ("INT", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "extend"
    CATEGORY = "短剧"

    def _prompt(self, motion, frame_index):
        motion = (motion or "").strip()
        turning = "转身" in motion or "离开" in motion
        if frame_index < self.FPS * 2:
            return "她面对镜头说话，头和肩膀只有很小的动作。不要转身，不要改变人物外貌和服装。"
        if turning:
            return "接着上一帧。她平稳转身，先侧身再背对镜头，动作连续，不要跳切，不要改变服装。"
        return f"接着上一帧，把动作做完，动作连续，不要跳切。{motion}"

    def extend(self, model, clip, vae, 画面, 动作, 宽, 高):
        target = self.FPS * self.SECONDS
        negative = CLIPTextEncode().encode(clip, self.NEGATIVE)[0]
        start = 画面[:1]
        pieces = []
        seed = 3
        while sum(part.shape[0] for part in pieces) < target:
            done = sum(part.shape[0] for part in pieces)
            text = self._prompt(动作, done)
            print(f"[短剧] 五秒成片 {done}/{target} 帧：{text}")
            positive = CLIPTextEncode().encode(clip, text)[0]
            latent = Wan22ImageToVideoLatent.execute(vae, int(宽), int(高), self.CHUNK, 1, start)[0]
            sampled = common_ksampler(model, seed, 20, 4.0, "euler", "simple", positive, negative, latent)[0]
            frames = VAEDecode().decode(vae, sampled)[0].detach().float().cpu()
            if pieces:
                frames = frames[1:]
            pieces.append(frames)
            start = frames[-1:].clone()
            seed += 1
            del latent, sampled
            model_management.soft_empty_cache()
        video = torch.cat(pieces, 0)[:target]
        print(f"[短剧] 五秒成片完成，{video.shape[0]} 帧。")
        return (video,)


class DuanjuVoice:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "对白": ("STRING", {"forceInput": True}),
                "说话人": ("STRING", {"forceInput": True}),
                "情绪": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "speak"
    CATEGORY = "短剧"

    def speak(self, images, 对白, 说话人, 情绪):
        _release()
        dest = Path(folder_paths.get_temp_directory()) / "duanju-line.wav"
        subprocess.run(
            [
                str(DUANJU_PYTHON),
                str(DUANJU),
                "说",
                "--文本",
                对白 or "",
                "--说话人",
                说话人 or "Vivian",
                "--情绪",
                情绪 or "",
                "--输出",
                str(dest),
            ],
            check=True,
        )
        with wave.open(str(dest), "rb") as handle:
            rate = handle.getframerate()
            channels = handle.getnchannels()
            frames = handle.readframes(handle.getnframes())
            width = handle.getsampwidth()
        if width == 2:
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            audio = np.frombuffer(frames, dtype=np.float32).copy()
        audio = audio.reshape(-1, channels).T
        waveform = torch.from_numpy(audio).unsqueeze(0)
        return ({"waveform": waveform, "sample_rate": rate},)


class DuanjuSubtitle:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "对白": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "burn"
    CATEGORY = "短剧"

    def burn(self, images, 对白):
        text = (对白 or "").strip()
        if not text:
            return (images,)
        frames = []
        font = ImageFont.truetype(FONT, 34, index=FONT_INDEX)
        for frame in images:
            array = (frame.detach().float().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            image = Image.fromarray(array).convert("RGBA")
            draw = ImageDraw.Draw(image)
            width, height = image.size
            max_width = width - 48
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
            top = height - 72 - block
            draw.rectangle((16, top - 12, width - 16, top + block + 8), fill=(0, 0, 0, 150))
            for index, line in enumerate(lines):
                line_width = draw.textlength(line, font=font)
                draw.text(((width - line_width) / 2, top + index * line_height), line, font=font, fill=(255, 255, 255, 255))
            rgb = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
            frames.append(torch.from_numpy(rgb))
        return (torch.stack(frames),)


class DuanjuExport:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "image": ("IMAGE",),
                "集": ("STRING", {"forceInput": True}),
                "镜号": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("路径",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "短剧"

    def save(self, video, image, 集, 镜号):
        from comfy_api.latest._util.video_types import VideoCodec, VideoContainer

        duanju = _duanju()
        folder = duanju.project_dir(集)
        shot = f"{int(镜号):02d}" if str(镜号).isdigit() else str(镜号)
        video_path = folder / "视频" / f"{shot}.mp4"
        image_path = folder / "图片" / f"{shot}.png"
        video.save_to(str(video_path), format=VideoContainer.MP4, codec=VideoCodec.H264)
        frame = image[0].detach().float().clamp(0, 1).cpu().numpy()
        Image.fromarray((frame * 255).astype("uint8")).save(image_path)
        combined = duanju.join_shots(folder)
        print(f"[短剧] 第 {shot} 镜已写入，合成：{combined}")
        return (str(combined),)


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
VIDEO_EXTS = (".mp4", ".mov", ".webm", ".m4v")


def _media_files(exts):
    folder = folder_paths.get_input_directory()
    names = ["无"]
    for name in sorted(os.listdir(folder)):
        if os.path.isfile(os.path.join(folder, name)) and name.lower().endswith(exts):
            names.append(name)
    return names


def _filled(name):
    text = str(name or "").strip()
    if not text or text == "无":
        return None
    return text


def _pick_slot(names, index):
    filled = [_filled(name) for name in names]
    if index >= 0 and index < len(filled) and filled[index]:
        return filled[index]
    for name in filled:
        if name:
            return name
    return None


def _open_image(name):
    path = folder_paths.get_annotated_filepath(name)
    return Image.open(path).convert("RGB")


def _cover(image, width, height):
    scale = max(width / image.width, height / image.height)
    resized = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.Resampling.LANCZOS)
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def _contain(image, width, height):
    scale = min(width / image.width, height / image.height)
    return image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.Resampling.LANCZOS)


def _tensor(image):
    array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def _video_frames(name, count=9):
    path = folder_paths.get_annotated_filepath(name)
    container = av.open(path)
    frames = []
    for frame in container.decode(container.streams.video[0]):
        frames.append(frame.to_image())
        if len(frames) >= 48:
            break
    container.close()
    if len(frames) > count:
        indexes = [round(i * (len(frames) - 1) / (count - 1)) for i in range(count)]
        frames = [frames[i] for i in indexes]
    return frames


class _RefFile:
    EXTS = IMAGE_EXTS
    VIDEO = False
    NAME = ""

    @classmethod
    def INPUT_TYPES(cls):
        option = {"video_upload": True} if cls.VIDEO else {"image_upload": True}
        return {"required": {cls.NAME: (_media_files(cls.EXTS), option)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("文件",)
    FUNCTION = "emit"
    CATEGORY = "短剧"

    def emit(self, **values):
        return (values[self.NAME],)


class DuanjuChar1(_RefFile):
    NAME = "主角参考图1"


class DuanjuChar2(_RefFile):
    NAME = "主角参考图2"


class DuanjuChar3(_RefFile):
    NAME = "主角参考图3"


class DuanjuBg1(_RefFile):
    NAME = "背景参考图1"


class DuanjuBg2(_RefFile):
    NAME = "背景参考图2"


class DuanjuVid1(_RefFile):
    EXTS = VIDEO_EXTS
    VIDEO = True
    NAME = "视频参考1"


class DuanjuVid2(_RefFile):
    EXTS = VIDEO_EXTS
    VIDEO = True
    NAME = "视频参考2"


class DuanjuComposeStart:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "画面": ("IMAGE",),
                "主角1": ("STRING", {"forceInput": True}),
                "主角2": ("STRING", {"forceInput": True}),
                "主角3": ("STRING", {"forceInput": True}),
                "背景1": ("STRING", {"forceInput": True}),
                "背景2": ("STRING", {"forceInput": True}),
                "视频1": ("STRING", {"forceInput": True}),
                "视频2": ("STRING", {"forceInput": True}),
                "角色序号": ("INT", {"forceInput": True}),
                "镜号": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("起始画面",)
    FUNCTION = "build"
    CATEGORY = "短剧"

    def build(self, 画面, 主角1, 主角2, 主角3, 背景1, 背景2, 视频1, 视频2, 角色序号, 镜号):
        frame = (画面[0].detach().float().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        base = Image.fromarray(frame).convert("RGB")
        width, height = base.size
        shot_index = max(int(镜号), 1) - 1 if str(镜号).isdigit() else 0
        video_name = _pick_slot((视频1, 视频2), shot_index)
        if video_name:
            frames = [_tensor(_cover(image.convert("RGB"), width, height)) for image in _video_frames(video_name)]
            if frames:
                print(f"[短剧] 这一镜用参考视频开头：{video_name}")
                return (torch.cat(frames, 0),)
        character_name = None if int(角色序号) < 0 else _pick_slot((主角1, 主角2, 主角3), int(角色序号))
        background_name = _pick_slot((背景1, 背景2), shot_index)
        if background_name:
            base = _cover(_open_image(background_name), width, height)
        if character_name:
            person = _contain(_open_image(character_name), int(width * 0.78), int(height * 0.78))
            plate = base.copy()
            plate.paste(person, ((width - person.width) // 2, (height - person.height) // 2))
            base = plate
        if character_name or background_name:
            print(f"[短剧] 起始画面使用参考图 主角={character_name or '无'} 背景={background_name or '无'}")
        return (_tensor(base),)


NODE_CLASS_MAPPINGS = {
    "DuanjuShot": DuanjuShot,
    "DuanjuChar1": DuanjuChar1,
    "DuanjuChar2": DuanjuChar2,
    "DuanjuChar3": DuanjuChar3,
    "DuanjuBg1": DuanjuBg1,
    "DuanjuBg2": DuanjuBg2,
    "DuanjuVid1": DuanjuVid1,
    "DuanjuVid2": DuanjuVid2,
    "DuanjuComposeStart": DuanjuComposeStart,
    "DuanjuZImageStack": DuanjuZImageStack,
    "DuanjuWanStack": DuanjuWanStack,
    "DuanjuSeconds": DuanjuSeconds,
    "DuanjuVoice": DuanjuVoice,
    "DuanjuSubtitle": DuanjuSubtitle,
    "DuanjuExport": DuanjuExport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DuanjuShot": "短剧本镜",
    "DuanjuChar1": "主角参考图1",
    "DuanjuChar2": "主角参考图2",
    "DuanjuChar3": "主角参考图3",
    "DuanjuBg1": "背景参考图1",
    "DuanjuBg2": "背景参考图2",
    "DuanjuVid1": "视频参考1",
    "DuanjuVid2": "视频参考2",
    "DuanjuComposeStart": "短剧起始画面",
    "DuanjuZImageStack": "短剧文生图模型",
    "DuanjuWanStack": "短剧图生视频模型",
    "DuanjuSeconds": "短剧五秒",
    "DuanjuVoice": "短剧配音",
    "DuanjuSubtitle": "短剧字幕",
    "DuanjuExport": "短剧归档",
}
