# TODO: Train code here
import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from model import DeepReceiver, compute_loss, filter_invalid_targets
from tqdm import tqdm  # 进度条工具
from dataset import create_dataloader


def train_model(data_path, batch_size=32, num_epochs=20, learning_rate=0.001, save_path="model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # 加载数据集
    train_loader, val_loader = create_dataloader(data_path, batch_size=batch_size, train_ratio=0.8)

    model = DeepReceiver().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 30)

        # 训练阶段
        model.train()
        train_loss = 0
        train_loader = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")  # 使用 tqdm 显示进度条
        for batch_idx, (iq_wave, symb_seq, symb_mask, symb_type, symb_wid) in enumerate(train_loader):
            iq_wave = iq_wave.permute(0, 2, 1).to(device)
            symb_seq = symb_seq.clamp(0, 1).to(device)

            optimizer.zero_grad()
            outputs = model(iq_wave)

            # 计算损失
            loss = compute_loss(outputs, symb_seq)
            loss.backward()
            optimizer.step()

            # 累加损失
            train_loss += loss.item()

            # 更新 tqdm 进度条信息
            train_loader.set_postfix({"Batch Loss": loss.item()})

        train_loss /= len(train_loader)
        print(f"Train Loss: {train_loss:.4f}")

        # 验证阶段
        model.eval()
        val_loss = 0
        val_loader = tqdm(val_loader, desc="Validating")  # 使用 tqdm 显示验证进度
        with torch.no_grad():
            for batch_idx, (iq_wave, symb_seq, symb_mask, symb_type, symb_wid) in enumerate(val_loader):
                iq_wave = iq_wave.permute(0, 2, 1).to(device)
                symb_seq = symb_seq.to(device)
                outputs = model(iq_wave)

                # 计算损失
                loss = compute_loss(outputs, symb_seq).item()
                val_loss += loss

                # 更新 tqdm 进度条信息
                val_loader.set_postfix({"Batch Loss": loss})

        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch + 1} with validation loss {val_loss:.4f}")


def evaluate_model(data_path, batch_size=32, model_path="model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader = create_dataloader(data_path, batch_size=batch_size, train_ratio=0.8)

    model = DeepReceiver().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    total_correct = 0
    total_bits = 0
    with torch.no_grad():
        for iq_wave, symb_seq, symb_mask, symb_type, symb_wid in val_loader:
            iq_wave = iq_wave.permute(0, 2, 1).to(device)
            symb_seq = symb_seq.to(device)
            outputs = model(iq_wave)

            # 忽略填充值
            filtered_outputs, filtered_targets = filter_invalid_targets(symb_seq, outputs)

            for output, target in zip(filtered_outputs, filtered_targets):
                predictions = torch.argmax(torch.softmax(output, dim=1), dim=1)
                total_correct += (predictions == target).sum().item()
                total_bits += target.numel()

    accuracy = total_correct / total_bits
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")



if __name__ == "__main__":
    # 获取数据路径
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    print(SCRIPT_DIR)
    data_path = os.path.join(SCRIPT_DIR, "train_data")
    print(data_path)

    # 设置训练参数
    batch_size = 32
    num_epochs = 20
    learning_rate = 0.001
    save_path = "deepreceiver_best.pth"

    # 训练模型
    print("start training")
    train_model(data_path, batch_size, num_epochs, learning_rate, save_path)
    print("end training")
    # 评估模型
    evaluate_model(data_path, batch_size, save_path)
