import random

import gradio as gr
from css.utils import *
import time

# 预置语音生成
def preset():
    # sound_choices = ['中文女', '中文男', '英文女', '英文男', '日语男', '粤语女', '韩语女']
    sound_choices = sft_spk
    def random_seed():
        return random.randint(1, 100000000)

    def generate_audio(_sound_radio, _synthetic_input_textbox, _seed):
        start_time = time.time()
        print(_sound_radio, _synthetic_input_textbox, _seed)
        if _synthetic_input_textbox == '':
            gr.Warning('合成文本为空，您是否忘记输入合成文本？')
            return (target_sr, default_data)
        set_all_random_seed(_seed)
        if use_instruct(_synthetic_input_textbox):
            model = cosyvoice_instruct
        else:
            model = cosyvoice
        # model = cosyvoice
        audio_tensors = []
        for output in model.inference_sft(_synthetic_input_textbox, _sound_radio):
            audio_tensors.append(output['tts_speech'])
            yield (target_sr, output['tts_speech'].numpy().flatten())
        # 将所有音频片段连接成一个单独的Tensor
        print("合成音频片段维度：{}".format(audio_tensors.shape))
        full_audio = torch.cat(audio_tensors, dim=1) if audio_tensors else torch.empty(0)

        audio_data = postprocess(full_audio).numpy().flatten()
        # print("检测结果：{}".format(detect_voice(audio_data, target_sr)))
        audio_data = add_watermark(audio_data)
        # print("检测结果：{}".format(detect_voice(audio_data, target_sr)))
        audio_data = (audio_data * (2**15)).astype(np.int16)
        # with open('{}/tmp/tts_output.wav'.format(os.getcwd()), 'wb') as f:
        #     f.write(audio_data)
        save_audio('tmp/tts_output.wav', audio_data)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"代码执行时间：{execution_time} 秒")
        return (target_sr, audio_data)
    def update_download_audio_button():
        return gr.DownloadButton(label="下载音频",value='tmp/tts_output.wav',variant="primary")
    def change_download_audio_button():
        return gr.DownloadButton(label="下载音频",variant="secondary")
    with gr.Column():
        sound_radio = gr.Radio(choices=sound_choices,
                               value=sound_choices[0],
                               label="选择预置音色")
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
                          inputs=[sound_radio, synthetic_input_textbox, seed],
                          outputs=[output_audio])
    output_audio.change(fn=update_download_audio_button, outputs=[download_audio_button])
    download_audio_button.click(fn=change_download_audio_button, outputs=[download_audio_button])