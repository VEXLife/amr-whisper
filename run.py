# 运行输入python run.py ./test_data ./result.csv
import os 
import sys
import torch
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration, LogitsProcessorList
from dataset import CustomSignalDataset
from model import SignalTokenizer, SignalFeatureExtractor, SignalLogitsProcessor
from vocab import vocab
from torch.nn.utils.rnn import pad_sequence as rnn_utils

def collator_fn(batch):
    # 过滤掉 batch 中的 None 数据项
    batch = [item for item in batch if item[0] is not None]
    
    if len(batch) == 0:
        raise ValueError("Batch contains only invalid data (None).")
    
    # 获取数据（信号特征）
    input_features = rnn_utils([item[0] for item in batch], batch_first=True)
    
    # 对于没有标签的任务，我们可以返回 None 或默认标签（如 0）
    labels = None  # 没有标签
    
    return {
        "input_features": input_features,
        "labels": labels,
    }

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

    # 检查设备加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = WhisperForConditionalGeneration.from_pretrained("./whisperiq.ckpt").to(device)
    model.eval()
    # 初始化 Tokenizer 和 logits_processor 和 dataset
    tokenizer = SignalTokenizer(vocab)
    logits_processor = SignalLogitsProcessor()
    logits_processor_list = LogitsProcessorList([logits_processor])
    dataset = CustomSignalDataset(testpath, SignalFeatureExtractor(2048), tokenizer)

    # 定义批量大小
    batch_size = 16  # 可以根据GPU显存调整，例如8或32
    # 使用 DataLoader 进行批量加载
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collator_fn)
    # 遍历dataloader，查看数据
    #for batch_idx, batch in enumerate(dataloader):
    #    print(f"Batch {batch_idx + 1}")
    #    print(f"Data: {batch['input_features']}")
    #    print(f"Labels: {batch['labels']}")

    total_files = len(test_file_lst)
    #print(f"Total files to process: {total_files}")
    processed_files = 0

    for batch_idx, batch in enumerate(dataloader):
        iq_wave_batch = batch['input_features'].to(device)
        with torch.no_grad():
            # 使用 generate 方法自动生成序列
            tk_batch = model.generate(
                iq_wave_batch,
                max_length=448,
                logits_processor=logits_processor_list,
                do_sample=False,  # 禁用采样，确保 logits_processor 生效
                num_beams=1       # 贪心解码，避免采样带来的随机性
            )
            #print(f"Generated tk_batch: {tk_batch}")  # 检查生成结果
        # 解码生成的序列
        s = tokenizer.batch_decode(tk_batch)
        #print(f"Decoded batch: {s}")  # 检查解码结果
        current_batch_size = iq_wave_batch.size(0)
        for i in range(current_batch_size):
            if processed_files >= total_files:
                break
            # 解析解码后的结果
            file_name = test_file_lst[processed_files]
            modulation_type = int(s[0][i])  # 如果是 Tensor，使用 .item()，否则直接转换
            symbol_width_tensor = float(s[1][i])  # 如果需要格式化为浮点数
            symbol_width = f"{symbol_width_tensor:.2f}"  # 格式化符号宽度
            code_sequence = s[2][i]  # 假设是一个可迭代的序列
            code_sequence_str = ' '.join(map(str, code_sequence))  # 将符号序列转换为字符串
            # 将结果添加到列表中
            result.append(f"{file_name},{modulation_type},{symbol_width},{code_sequence_str}")
            processed_files += 1
        print(f"Processed {processed_files}/{total_files} files.")
    # 将预测结果保存到 result_save_path
    with open(result_save_path, 'w') as f:
        f.write('\n'.join(result))


if __name__ == "__main__":
    to_pred_dir = sys.argv[1]  # 不可更改！
    result_save_path = sys.argv[2]  # 不可更改！
    main(to_pred_dir, result_save_path)  # 不可更改！
