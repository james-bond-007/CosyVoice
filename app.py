import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append('{}/third_party/AcademiCodec'.format(ROOT_DIR))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))

# from modelscope import snapshot_download
# snapshot_download('speech_tts/speech_kantts_ttsfrd', revision='v1.0.3', allow_file_pattern='ttsfrd-0.3.6-cp38-cp38-linux_x86_64.whl', local_dir='pretrained_models/speech_kantts_ttsfrd')
# os.system('cd pretrained_models/speech_kantts_ttsfrd/ && pip install ttsfrd-0.3.6-cp38-cp38-linux_x86_64.whl')
# os.system('sed -i s@pydantic.typing@typing_extensions@g .conda/lib/python3.8/site-packages/inflect/__init__.py')
# os.system('sed -i s@https://huggingface.co/facebook/audioseal/resolve/main/generator_base.pth@{}@g .conda/lib/python3.8/site-packages/audioseal/cards/audioseal_wm_16bits.yaml'.format(os.path.join(ROOT_DIR, 'pretrained_models/audioseal/generator_base.pth')))

import gradio as gr
from css.advanced import advanced
from css.custom import custom
from css.preset import preset

audio_mode_choices = [('预置语音生成', 'preset'), ('定制语音生成（复刻录制声音）', 'custom'),
                      ('高级语音生成（自然语言控制）', 'advanced')]

from css.utils import instruct_dict, gt_host_ip
import darkdetect


custom_css = """
.full-height {
    height: 100%;
}
"""



default_layout = 'custom'
def main():
    def on_audio_mode_change(_audio_mode_radio):
        print(_audio_mode_radio)
        # gr.Text(label="操作提示",value=instruct_dict[_audio_mode_radio],)
        yield {
            preset_layout: gr.update(visible=_audio_mode_radio == 'preset'),
            custom_layout: gr.update(visible=_audio_mode_radio == 'custom'),
            advanced_layout: gr.update(visible=_audio_mode_radio == 'advanced'),
            operation_steps: gr.update(value=instruct_dict[_audio_mode_radio])
        }
    
    with gr.Blocks(css=custom_css) as demo:
        gr.set_static_paths(paths=["logo.png", "logo1.png"])
        if darkdetect.isDark():
            img_text = '/file=logo1.png'
        else:
            img_text = '/file=logo.png'
        gr.HTML('<img src="'+img_text+'" width="50%">')
        audio_mode_radio = gr.Radio(choices=audio_mode_choices,
                                    value=default_layout,
                                    label="选择语音生成模式")
        operation_steps = gr.Text(label="操作提示",
            value=instruct_dict[default_layout],)
        with gr.Row():
            with gr.Column(visible=default_layout == 'preset') as preset_layout:
                preset()
            with gr.Column(visible=default_layout == 'custom') as custom_layout:
                custom()
            with gr.Column(visible=default_layout == 'advanced') as advanced_layout:
                advanced()

        audio_mode_radio.change(
            fn=on_audio_mode_change,
            inputs=[audio_mode_radio],
            outputs=[preset_layout, custom_layout, advanced_layout,operation_steps])

    demo.queue(max_size=4, default_concurrency_limit=2)
    # demo.launch(server_name=gt_host_ip(), server_port=5000)
    demo.launch(server_name="0.0.0.0", server_port=5000)

if __name__ == '__main__':
    main()
