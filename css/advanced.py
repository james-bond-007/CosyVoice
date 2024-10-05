import random

import gradio as gr
from css.utils import *
import time

# 高级语音生成
def advanced():

    sound_choices = ['中文女', '中文男', '日语男', '英文女', '英文男', '粤语女', '韩语女']
    # sound_choices = sft_spk
    def random_seed():
        return random.randint(1, 100000000)

    def generate_audio(_sound_radio, _speech_status_textbox,
                       _synthetic_input_textbox, _seed):
        start_time = time.time()
        print(_sound_radio, _speech_status_textbox, _synthetic_input_textbox, _seed)
        if _synthetic_input_textbox == '':
            gr.Warning('合成文本为空，您是否忘记输入合成文本？')
            return (target_sr, default_data)
        set_all_random_seed(_seed)
        model = CosyVoice('pretrained_models/CosyVoice-300M-Instruct')
        # output = model.inference_instruct(_synthetic_input_textbox, _sound_radio, _speech_status_textbox)
        audio_tensors = []
        for output in model.inference_instruct(_synthetic_input_textbox, _sound_radio, _speech_status_textbox):
            audio_tensors.append(output['tts_speech'])
            yield (target_sr, output['tts_speech'].numpy().flatten())
        # print("合成音频数组shape：{}".format(audio_tensors.shape))
        print("合成音频：{} ".format(audio_tensors))
        # 将所有音频片段连接成一个单独的Tensor
        full_audio = torch.concat(audio_tensors, dim=1)
        # print("合成音频shape：{} ".format(full_audio.shape))
        print("合成音频：{} ".format(full_audio))

        audio_data = postprocess(full_audio).numpy().flatten()
        # print("检测结果：{}".format(detect_voice(audio_data, target_sr)))
        audio_data = add_watermark(audio_data)
        # print("检测结果：{}".format(detect_voice(audio_data, target_sr)))
        audio_data = (audio_data * (2**15)).astype(np.int16)
        save_audio('tmp/instruct_output.wav', audio_data)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"代码执行时间：{execution_time} 秒")
        return (target_sr, audio_data)
    def update_download_audio_button():
        return gr.DownloadButton(label="下载音频",value='tmp/instruct_output.wav', variant="primary")
    def change_download_audio_button():
        return gr.DownloadButton(label="下载音频",variant="secondary")
    with gr.Column():
        sound_radio = gr.Radio(choices=sound_choices,
                               value=sound_choices[0],
                               label="选择预置音色")
    with gr.Column():
        speech_status_textbox = gr.Textbox(label="描述语音状态")
        gr.Examples(
            label="示例控制文本",
            examples=[["Theo 'Crimson', is a fiery, passionate rebel leader. Fights with fervor for justice, but struggles with impulsiveness."],
                ["Selene 'Moonshade', is a mysterious, elegant dancer with a connection to the night. Her movements are both mesmerizing and deadly. "],
                ["A female speaker with normal pitch, slow speaking rate, and sad emotion."],
            ],
            inputs=[speech_status_textbox])
    with gr.Column():
        synthetic_input_textbox = gr.Textbox(label="输入合成文本",
                                             value="大家好，我是河北广播电视台虚拟主播冀小佳。")
        gr.Examples(
            label="示例文本",
            examples=example_tts_text,
            inputs=[synthetic_input_textbox])

    with gr.Accordion(label="随机种子"):
        with gr.Row():
            with gr.Column(scale=1, min_width=180):
                seed_button = gr.Button(value="\U0001F3B2 随机换一换",
                                        elem_classes="full-height")
            with gr.Column(scale=10):
                seed = gr.Number(show_label=False,
                                 value=0,
                                 container=False,
                                 elem_classes="full-height")
    with gr.Column():
        generate_button = gr.Button("生成音频", variant="primary", size="lg")

    with gr.Column():
        output_audio = gr.Audio(label="合成音频",
                                streaming=True,
                                autoplay=True,
                                show_download_button=False)
    download_audio_button = gr.DownloadButton(label="下载音频")
    seed_button.click(fn=random_seed, outputs=[seed])
    generate_button.click(fn=generate_audio,
                          inputs=[
                              sound_radio, speech_status_textbox,
                              synthetic_input_textbox, seed
                          ],
                          outputs=[output_audio])
    output_audio.change(fn=update_download_audio_button, outputs=[download_audio_button])
    download_audio_button.click(fn=change_download_audio_button, outputs=[download_audio_button])
