# 运行输入python run.py ./test_data ./result.csv
# 在model = LitDenseNet.load_from_checkpoint(
#       'checkpoints/end-epoch=0.ckpt',
#       max_bits=2000
#   )修改ckpt文件
import os
import sys
import pandas as pd
import torch
from model import LitDenseNet
from dataset import recover_symb_seq_from_bin_seq, get_modulation_symb_bits

def main(to_pred_dir, result_save_path):
    """
    主函数，用于执行脚本的主要逻辑。

    参数：
        to_pred_dir: 测试集文件夹上层目录路径，不可更改!
        result_save_path: 预测结果文件保存路径，官方已指定为csv格式，不可更改!
    """
    # 获取测试集文件夹路径，不可更改！
    testpath = os.path.abspath(to_pred_dir)

    # 获取测试集文件列表
    test_file_lst = []
    for root, _, files in os.walk(testpath):  # 遍历所有子文件夹
        for name in files:
            if name.endswith('.csv'):
                test_file_lst.append(os.path.join(root, name))

    # 初始化结果文件，定义表头，注意逗号之间无空格
    result = ['file_name,modulation_type,symbol_width,code_sequence']

    # 加载训练好的模型（模型路径根据实际情况调整）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LitDenseNet.load_from_checkpoint(
        'checkpoints/end-epoch=0.ckpt',
        max_bits=2000  # 替换为训练时的 max_bits 值
    )
    model.eval().to(device)

    # 循环测试集文件列表对每个文件进行预测
    for filename in test_file_lst:
        # 待预测文件路径
        filepath = os.path.join(testpath, filename)

        # 读入测试文件，这里的测试文件为无表头的两列信号值
        df = pd.read_csv(filepath, header=None, names=['I', 'Q'])

        # 转换为模型输入的格式
        iq_wave = torch.tensor(df.values, dtype=torch.float32).unsqueeze(0).to(device)  # 添加batch维度
        iq_wave = iq_wave.permute(0, 2, 1)  # 调整为 (batch, channel, time)

        # 模型预测
        with torch.no_grad():
            output_bits_hat, symb_type, symbol_width = model.predict_step(iq_wave, batch_idx=0)

        # 解码预测结果
        modulation_type = symb_type.item() + 1
        symbol_width = symbol_width.item()
        symb_bits = get_modulation_symb_bits(modulation_type)
        code_sequence = recover_symb_seq_from_bin_seq(output_bits_hat[0], symb_bits, device).squeeze(0).cpu().numpy()
        code_sequence_str = ' '.join(map(str, code_sequence.astype(int)))

        # 添加结果到列表
        result.append(f"{filename},{modulation_type},{symbol_width},{code_sequence_str}")

    # 将预测结果保存到result_save_path
    with open(result_save_path, 'w') as f:
        f.write('\n'.join(result))

if __name__ == "__main__":
    # ！！！以下内容不允许修改，修改会导致评分出错
    to_pred_dir = sys.argv[1]  # 官方给出的测试文件夹上层的路径，不可更改！
    result_save_path = sys.argv[2]  # 官方给出的预测结果保存文件路径，已指定格式为csv，不可更改！
    main(to_pred_dir, result_save_path)  # 运行main脚本，入参只有to_pred_dir, result_save_path，不可更改！