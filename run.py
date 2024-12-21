# 运行输入python run.py ./test_data ./result.csv
import os
import sys
import pandas as pd
import torch
import lightning.pytorch as L
from dataset import SignalDataModule, recover_symb_seq_from_bin_seq, get_modulation_symb_bits
from model import LitDenseNet


def main(to_pred_dir, result_save_path):
    """
    主函数，用于执行脚本的主要逻辑。

    参数：
        to_pred_dir: 测试集文件夹上层目录路径，不可更改！
        result_save_path: 预测结果文件保存路径，官方已指定为csv格式，不可更改！
    """
    # 获取测试集文件夹路径，不可更改！
    testpath = os.path.join(os.path.abspath(to_pred_dir), 'test')

    # 获取测试集文件列表
    test_file_lst = [name for name in os.listdir(testpath) if name.endswith('.csv')]

    # 初始化结果文件，定义表头，注意逗号之间无空格
    result = ['file_name,modulation_type,symbol_width,code_sequence']

    # 加载模型
    model = LitDenseNet.load_from_checkpoint(
        'best.ckpt'
    )
    model.eval()

    # 使用 SignalDataModule 加载测试数据
    data_module = SignalDataModule(
        data_path=testpath,
        batch_size=32,  # 根据需求调整 batch size
        num_workers=3
    )

    # 初始化 Trainer
    trainer = L.Trainer()

    # 使用 Trainer 的 predict 方法进行预测
    predictions = trainer.predict(model, datamodule=data_module)

    for filename, pred in zip(test_file_lst, predictions):
        output_bits_hat, symb_type_batch, symbol_width_batch = pred
        # 遍历批量数据
        for idx in range(symb_type_batch.shape[0]):
            modulation_type = symb_type_batch[idx].item() + 1
            symbol_width = symbol_width_batch[idx].item()

            # 解码预测结果
            symb_bits = get_modulation_symb_bits(modulation_type)
            code_sequence = recover_symb_seq_from_bin_seq(
                output_bits_hat[idx], symb_bits, device=torch.device("cpu")
            )
            code_sequence_str = ' '.join(map(str, code_sequence.squeeze(0).numpy().astype(int)))

            # 添加结果到列表
            result.append(f"{filename},{modulation_type},{symbol_width: .2f},{code_sequence_str}")


    # 将预测结果保存到 result_save_path
    with open(result_save_path, 'w') as f:
        f.write('\n'.join(result))


if __name__ == "__main__":
    # ！！！以下内容不允许修改，修改会导致评分出错
    to_pred_dir = sys.argv[1]  # 官方给出的测试文件夹上层的路径，不可更改！
    result_save_path = sys.argv[2]  # 官方给出的预测结果保存文件路径，已指定格式为csv，不可更改！
    main(to_pred_dir, result_save_path)  # 运行main脚本，入参只有to_pred_dir, result_save_path，不可更改！