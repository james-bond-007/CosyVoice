
# import logging
# logging.getLogger("matplotlib").setLevel(logging.WARNING)
# logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',)

import os
import sys
import argparse
import gradio as gr
import numpy as np
import torch
import torchaudio
import random
import librosa as sf
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav, logging
from cosyvoice.utils.common import set_all_random_seed
import socket
from audioseal import AudioSeal
seal = AudioSeal.load_generator("audioseal_wm_16bits")
seal.eval()
detector = AudioSeal.load_detector("audioseal_detector_16bits")

from cosyvoice.cli.cosyvoice import CosyVoice
cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M')
cosyvoice_sft = CosyVoice('pretrained_models/CosyVoice-300M-SFT')
cosyvoice_instruct = CosyVoice('pretrained_models/CosyVoice-300M-Instruct')
sft_spk = cosyvoice_sft.list_avaliable_spks()
example_tts_text = ["我们走的每一步，都是我们策略的一部分；你看到的所有一切，包括我此刻与你交谈，所做的一切，所说的每一句话，都有深远的含义。",
                    "那位喜剧演员真有才，[laughter]一开口就让全场观众爆笑。",
                    "他搞的一个恶作剧，让大家<laughter>忍俊不禁</laughter>。",
                    "在面对挑战时，他展现了非凡的<strong>勇气</strong>与<strong>智慧</strong>。",
                    "你好，今日你食咗饭未呀？",
                    "As China strives for technological innovation represented by intelligent connected vehicles and autonomous driving",
                    "저는 이번 주에 친구와 함께 여행을 갈 계획입니다. ",
                    "こんにちは、いかがしていますか？"]

example_prompt_text = ["大家好，我是河北广播电视台虚拟主播冀小佳。",
                        "Hello everyone, I am Jixiaojia, a virtual host of Hebei Broadcasting Station."]

instruct_dict = {'preset': '1. 选择预训练音色。\n2. 输入合成文本或选择对应语言的示例文本。\n3. 点击生成音频按钮。',
                 'custom': '1. 在示例prompt音频文件中选择prompt音频文件，或选择示例侍录制文本录制prompt音频。若同时提供，prompt音频文件优先。\n2. 选择同语种或者跨语种，然后选择对应的示例文本。\n3. 点击生成音频按钮。',
                 'advanced': '1. 选择预训练音色。\n2. 输入instruct文本或者选择示例控制文本。\n3. 输入合成文本或者选择示例文本。\n4. 点击生成音频按钮'}
#加过了音频水印，target_sr 16000
prompt_sr, target_sr = 16000, 22050
default_data = np.zeros(target_sr)
def set_all_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

max_val = 0.8
def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = sf.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1,int(target_sr * 0.2))],dim=1)
    return speech

instruct_symbols = ['<endofprompt>', '<laughter>', '</laughter>',  '<strong>', '</strong>', '[laughter]', '[breath]']
def use_instruct(text):
    for symbol in instruct_symbols:
        if symbol in text:
            return True
    return False

def gt_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        print("本机IP地址为：" + ip)
    finally:
        s.close()
    return ip
@torch.inference_mode()
def add_watermark(waveform):
    waveform = torch.from_numpy(waveform).view(1,-1)
    # waveform_16k = torchaudio.transforms.Resample(orig_freq=22050, new_freq=16000)(waveform).view(1,-1)
    # print(waveform.shape)
    if waveform.dim() != 3:
        waveform = waveform.unsqueeze(1)  # 假设 audio_data 的形状是 (B, T)
        # print(waveform.shape)  # 应该是 (B, 1, T)
        # raise ValueError(f"Expected input to be 3-dimensional, got {waveform_16k.dim()} dimensions")
    watermark = seal.get_watermark(waveform, target_sr)
    waveform += watermark
    return waveform.flatten().numpy()

def add_noise(waveform, noise_level=0.005):
    noise = np.random.normal(0, noise_level, waveform.shape[0])
    return waveform + noise

def detect_voice(watermarked_audio, sr):
    # detector = AudioSeal.load_detector(("audioseal_detector_16bits"))
    watermarked_audio = torch.from_numpy(watermarked_audio).view(1,-1)
    # print(watermarked_audio.shape)
    if watermarked_audio.dim() != 3:
        watermarked_audio = watermarked_audio.unsqueeze(1)  # 假设 audio_data 的形状是 (B, T)
        # print(watermarked_audio.shape)  # 应该是 (B, 1, T)
    result, message = detector.detect_watermark(watermarked_audio, sample_rate=sr, message_threshold=0.5)
    print(f"\nThis is likely a watermarked audio: {result}")
    return result

def save_audio(path, audio_data):
    # sf.write(path, audio_data, sr=target_sr)
    # 将NumPy数组转换为PyTorch Tensor
    audio_data_tensor = torch.from_numpy(audio_data).view(1,-1)
    torchaudio.save(path, audio_data_tensor, sample_rate=target_sr)
