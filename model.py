# TODO: Model declarations here
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(BasicBlock, self).__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU()
        self.conv = nn.Conv1d(in_channels, growth_rate, kernel_size=5, padding=2)

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        out = torch.cat([x, out], dim=1)
        return out


class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionBlock, self).__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out = self.pool(self.conv(self.relu(self.bn(x))))
        return out


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList([
            BasicBlock(in_channels + i * growth_rate, growth_rate) for i in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DeepReceiver(nn.Module):
    def __init__(self, num_bits=150):
        super(DeepReceiver, self).__init__()
        self.initial_conv = nn.Conv1d(2, 64, kernel_size=5, padding=2)  # Input has 2 channels (Re, Im)

        # Define DenseNet structure
        self.transition1 = TransitionBlock(64, 128)
        self.dense1 = DenseBlock(2, 128, 128)

        self.transition2 = TransitionBlock(128 + 2 * 128, 64)
        self.dense2 = DenseBlock(3, 64, 64)

        self.transition3 = TransitionBlock(64 + 3 * 64, 64)
        self.dense3 = DenseBlock(4, 64, 64)

        self.transition4 = TransitionBlock(64 + 4 * 64, 64)
        self.dense4 = DenseBlock(3, 64, 64)

        self.final_conv = nn.Conv1d(64 + 3 * 64, num_bits, kernel_size=5, padding=2)

        # Global Pooling layers
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Output layers for binary classifiers
        self.binary_classifiers = nn.ModuleList([nn.Linear(num_bits * 2, 2) for _ in range(num_bits)])

    def forward(self, x):
        # Initial convolution
        x = self.initial_conv(x)

        # DenseNet layers
        x = self.transition1(x)
        x = self.dense1(x)

        x = self.transition2(x)
        x = self.dense2(x)

        x = self.transition3(x)
        x = self.dense3(x)

        x = self.transition4(x)
        x = self.dense4(x)

        # Final convolution
        x = self.final_conv(x)

        # Global pooling
        max_pooled = self.global_max_pool(x).squeeze(-1)
        avg_pooled = self.global_avg_pool(x).squeeze(-1)

        # Concatenate pooling results
        features = torch.cat([max_pooled, avg_pooled], dim=1)

        # Binary classification for each bit
        outputs = []
        for classifier in self.binary_classifiers:
            outputs.append(classifier(features))

        return outputs

def filter_invalid_targets(labels, outputs, ignore_index=-1):
    """
    过滤目标值，忽略填充值 (-1)。
    参数:
        labels: [batch_size, sequence_length]，目标值
        outputs: list，包含每个位的模型输出，每个元素形状为 [batch_size, 2]
        ignore_index: int，表示需要忽略的填充值
    返回:
        filtered_outputs: list，过滤后的有效输出
        filtered_targets: list，过滤后的有效目标值
    """
    valid_mask = labels != ignore_index  # 创建有效掩码
    filtered_outputs = []
    filtered_targets = []

    for i, output in enumerate(outputs):  # 遍历每个位的输出
        target = labels[:, i]  # 当前位的目标值
        valid_output = output[valid_mask[:, i]]  # 应用掩码过滤无效输出
        valid_target = target[valid_mask[:, i]]  # 应用掩码过滤无效目标

        if valid_output.shape[0] > 0:  # 如果存在有效值
            filtered_outputs.append(valid_output)
            filtered_targets.append(valid_target)

    return filtered_outputs, filtered_targets



# Loss function
def compute_loss(outputs, labels, ignore_index=-1):
    """
    计算损失，忽略填充值 (-1)。
    参数:
        outputs: list，每个位的模型输出，每个元素形状为 [batch_size, 2]
        labels: [batch_size, sequence_length]，目标值
        ignore_index: int，表示需要忽略的填充值
    返回:
        loss: float，总损失
    """
    criterion = nn.CrossEntropyLoss()
    loss = 0

    # 过滤掉无效的目标值
    filtered_outputs, filtered_labels = filter_invalid_targets(labels, outputs, ignore_index=ignore_index)

    for output, label in zip(filtered_outputs, filtered_labels):
        loss += criterion(output, label)  # 计算损失
    
    return loss



# Training step
def train_step(model, data_loader, optimizer):
    model.train()
    total_loss = 0
    for iq_signals, labels in data_loader:
        iq_signals, labels = iq_signals.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(iq_signals)
        loss = compute_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)


# Inference step
def inference(model, iq_signals):
    model.eval()
    iq_signals = iq_signals.cuda()
    with torch.no_grad():
        outputs = model(iq_signals)
        predictions = []
        for output in outputs:
            predictions.append(torch.argmax(F.softmax(output, dim=1), dim=1))
    return torch.stack(predictions, dim=1)
