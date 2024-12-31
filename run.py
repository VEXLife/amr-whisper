# 运行输入python run.py ./test_data ./result.csv
import os
import sys
import torch
from transformers import WhisperForConditionalGeneration, LogitsProcessorList
from dataset import SignalDataset, CustomSignalDataset
from model import SignalTokenizer, SignalFeatureExtractor, SignalLogitsProcessor
from vocab import vocab
    

def main(to_pred_dir, result_save_path):
    """
    主函数，用于执行脚本的主要逻辑。

    参数：
        to_pred_dir: 测试集文件夹上层目录路径，不可更改！
        result_save_path: 预测结果文件保存路径，官方已指定为csv格式，不可更改！
    """
    testpath = os.path.join(os.path.abspath(to_pred_dir), 'test')  # 不可更改！
    test_file_lst = [name for name in os.listdir(testpath) if name.endswith('.csv')]  # 不可更改！
    result = ['file_name,modulation_type,symbol_width,code_sequence']  # 不可更改！

    # 加载模型
    model = WhisperForConditionalGeneration.from_pretrained("./whisperiq.ckpt")  # 模型checkpoint解压到这个文件夹里
    model.eval()

    # 初始化 Tokenizer 和 Dataset
    tokenizer = SignalTokenizer(vocab)
    dataset = CustomSignalDataset(testpath, SignalFeatureExtractor(2048), tokenizer)
    
    logits_processor = SignalLogitsProcessor()
    logits_processor_list = LogitsProcessorList([logits_processor])

    # 遍历测试集文件，生成预测结果
    for idx, file_name in enumerate(test_file_lst):
        iq_wave = dataset[idx][0].unsqueeze(0)  # 添加 batch 维度
        tk = model.generate(iq_wave, max_length=448, logits_processor=logits_processor_list)  # 推理
        s = tokenizer.batch_decode(tk)
        modulation_type = int(s[0][0].item())
        symbol_width_tensor = s[1][0]
        symbol_width = f"{symbol_width_tensor:.2f}"
        code_sequence = s[2][0]
        code_sequence_str = ' '.join(map(str, code_sequence))
        result.append(f"{file_name},{modulation_type},{symbol_width},{code_sequence_str}")
    # 将预测结果保存到 result_save_path
    with open(result_save_path, 'w') as f:
        f.write('\n'.join(result))


if __name__ == "__main__":
    to_pred_dir = sys.argv[1]  # 不可更改！
    result_save_path = sys.argv[2]  # 不可更改！
    main(to_pred_dir, result_save_path)  # 不可更改！