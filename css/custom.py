import random

import gradio as gr
from css.utils import *
import soundfile as sf
import time

# 定制语音生成
def custom():

    def random_seed():
        return random.randint(1, 100000000)

    def generate_audio(_prompt_wav_upload, _prompt_wav_record, _prompt_input_textbox, _language_radio,
                       _synthetic_input_textbox, _seed):
        start_time = time.time()
        if _prompt_wav_upload is not None:
            _recorded_audio = _prompt_wav_upload
        elif _prompt_wav_record is not None:
            _recorded_audio = _prompt_wav_record
        else:
            _recorded_audio = None
        print(_recorded_audio, _prompt_input_textbox, _language_radio, _synthetic_input_textbox, _seed)
        if _synthetic_input_textbox == '':
            gr.Warning('合成文本为空，您是否忘记输入合成文本？')
            return (target_sr, default_data)
        set_all_random_seed(_seed)
        # if use_instruct(_synthetic_input_textbox):
        #     model = cosyvoice_instruct
        # else:
        #     model = cosyvoice_sft
        # model = cosyvoice_instruct
        prompt_speech_16k = postprocess(load_wav(_recorded_audio, prompt_sr))
        print("语言：{}".format(_language_radio))
        audio_tensors = []
        if _language_radio == 'cross' or _prompt_input_textbox == '':
            model = CosyVoice('pretrained_models/CosyVoice-300M')
            # output = model.inference_cross_lingual(_synthetic_input_textbox, prompt_speech_16k)
            for output in model.inference_cross_lingual(_synthetic_input_textbox, prompt_speech_16k):
                audio_tensors.append(output['tts_speech'])
                yield (target_sr, output['tts_speech'].numpy().flatten())
        else:
            model = CosyVoice('pretrained_models/CosyVoice-300M-Instruct')
            # output = model.inference_zero_shot(_synthetic_input_textbox, _prompt_input_textbox, prompt_speech_16k)
            for output in model.inference_zero_shot(_synthetic_input_textbox, _prompt_input_textbox, prompt_speech_16k):
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
        save_audio('tmp/clone_output.wav', audio_data)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"代码执行时间：{execution_time} 秒")
        return (target_sr, audio_data)
    def load_prompt_audio(file):
        audio_data, sr = sf.read(file)
        return (sr, audio_data)
    def prompt_audio_text(audio_path):
        # 处理音频文件的逻辑
        # ...
        # audio_data, sr = soundfile.read(audio_path)
        # audio_path = "/Users/loukuofeng/PycharmProjects/GPT-SoVITS/output/slicer_opt/24 标记 23.wav_0000000000_0000118080.wav"
        # result = funasr_asr.only_asr(audio_path)  # 假设处理后的音频文件路径为audio_path
        filename_without_extension, extension = os.path.splitext(audio_path)
        file_name = filename_without_extension.split("/")[-1]
        print("文件名（无扩展名）:", file_name)
        print("扩展名:", extension)
        return file_name
    def update_download_audio_button():
        return gr.DownloadButton(label="下载音频",value='tmp/clone_output.wav', variant="primary")
    def change_download_audio_button():
        return gr.DownloadButton(label="下载音频",variant="secondary")
        
    prompt_audio_path = "./prompt_audio/二零二四年是京津冀协同发展战略实施十周年.wav"

    with gr.Column():
        with gr.Row():
            with gr.Column():
                prompt_wav_record = gr.Audio(sources=['microphone'],
                                              label="录制音频文件",
                                              type='filepath')
                gr.Text("请点击录制，并朗读下方待录制文本（中文或英文）完成录入",
                            max_lines=1,
                            container=False,
                            interactive=False)
                prompt_input_textbox = gr.Textbox(label="输入待录制文本",
                                                  value=prompt_audio_text(prompt_audio_path),)
                gr.Examples(
                    label="示例待录制文本",
                    examples=example_prompt_text,
                    inputs=[prompt_input_textbox])
            with gr.Column():
                prompt_wav_upload = gr.Audio(sources='upload', 
                                             value=load_prompt_audio(prompt_audio_path), 
                                             type='filepath', 
                                             label='选择prompt音频文件，注意采样率不低于16khz')
                import random
                examples_list = [os.path.join(os.getcwd(), os.path.normpath(r"prompt_audio/"+f)) 
                                for f in os.listdir("prompt_audio/") if not f.startswith('.')]
                # examples_list = random.shuffle(examples_list)
                gr.Examples(
                    label="示例prompt音频文件",
                    examples=examples_list,
                    inputs=prompt_wav_upload,
                    outputs=prompt_input_textbox ,
                    fn=prompt_audio_text,
                    cache_examples=True,
                )

                    
            

    with gr.Column():
        language_radio = gr.Radio(choices=[('同语种', 'same'), ('跨语种', 'cross')],
                                  value='same',
                                  label="输入合成文本")
        synthetic_input_textbox = gr.Textbox(show_label=False,
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
    generate_button.click(
        fn=generate_audio,
        inputs=[prompt_wav_upload, prompt_wav_record, prompt_input_textbox, language_radio, synthetic_input_textbox, seed],
        outputs=[output_audio])
    output_audio.change(fn=update_download_audio_button, outputs=[download_audio_button])
    download_audio_button.click(fn=change_download_audio_button, outputs=[download_audio_button])
