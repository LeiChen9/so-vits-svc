# Code read note

## pipeline
### train.py
main函数：先统计机器信息（有几个卡），然后多卡多进程跑run函数
run函数：
1. 如果是第一轮，设置logger和writer
2. 设置好环境超参（process、seed、device）
3. 实例化TextAudioCollate
4. 通过files实例化TextAudioSpeakerLoader
5. 通过SynthesizerTrn实例化net_g

### data_utils.py
TextAudioCollate:
一个__call__函数：
传入一个torch tensor batch
根据c_len做排序
获取max_c_len, max_wav_len, 整个batch的长度lengths
生成各种shape的pad，全部赋0
对batch做padding

TextAudioSpeakerLoader
__init__: 获取各种超参数，设置seed，shuffle data，对文件做for读取，把get_audio的返回值放到一个list里面，叫self.cache
__getitem__: 对cache做random_slice
get_audio:
1. load_wav_to_torch
2. 对audio做基于max归一化和拉平
3. 对已经解析好的file做序列化，下次可以直接用
4. 对没解析好的做spectrogram_torch解析
5. load f0, uv, 做torch tensor
6. 如果有vol_embed的话，load它，成为volume变量
7. 获取lmin，对所有要返回的值做截断
8. return c, f0, spec, audio_norm, spk, uv, volume
random_slice
看不太懂

### mel_processing.py
spectrogram_torch：解析出来mel频谱

### models.py
SynthesizerTrn：
win test