#0317-改进的最终代码，可以替换混淆矩阵进行实验
# 融合后Accuracy: 0.9341
# 融合结果 Hamming_loss:0.0659, MAP:0.6826, AUC: 0.8852, F1: 0.6617, Precision: 0.8233, Recall: 0.6102


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import hamming_loss, roc_auc_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import os
from torchvision.models import ResNet50_Weights
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import torch.nn.functional as F



# 提取图像特征并保存到文件
def extract_and_save_features(data, image_folder, transform, output_path):
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model = nn.Sequential(*list(model.children())[:-1])  # 去掉最后的分类层
    model.eval()

    features_dict = {}
    with torch.no_grad():
        for idx, row in data.iterrows():
            image_path = os.path.join(image_folder, row['Image_ID'])
            print(f"提取特征: {image_path}")
            try:
                image = Image.open(image_path).convert('RGB')
                image = transform(image).unsqueeze(0)  # 添加 batch 维度
                features = model(image).squeeze().numpy()
                features_dict[row['Image_ID']] = features
            except Exception as e:
                print(f"跳过图像 {image_path}: {e}")

    # 保存提取的特征
    np.save(output_path, features_dict)
    print(f"特征已保存到 {output_path}")


# 自定义数据集类
class FusionDataset(Dataset):
    def __init__(self, data, feature_file):
        self.data = data.fillna(0)  # 填充数据集中的空值为0
        self.features = np.load(feature_file, allow_pickle=True).item()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_id = row['Image_ID']

        # 加载图像特征
        image_features = self.features.get(image_id, np.zeros(2048))

        expert_labels = row[['Fracture_Expert_Label', 'Pneumothorax_Expert_Label',
                             'Airspace_Opacity_Expert_Label', 'Nodule_Or_Mass_Expert_Label']].values.astype(float)
        ml_predictions = row[['Fracture_ML_Label', 'Pneumothorax_ML_Label',
                              'Airspace_Opacity_ML_Label', 'Nodule_Or_Mass_ML_Label']].values.astype(float)
        true_labels = row[['Fracture_GT_Label', 'Pneumothorax_GT_Label',
                           'Airspace_Opacity_GT_Label', 'Nodule_Or_Mass_GT_Label']].values.astype(float)
        

        # print(f"image_features.shape: {image_features.shape}")
        # print(f"expert_labels.shape: {expert_labels.shape}")
        # print(f"ml_predictions.shape: {ml_predictions.shape}")

        return {
            'image_features': torch.tensor(image_features, dtype=torch.float32),
            'expert_labels': torch.tensor(expert_labels, dtype=torch.float32),
            'ml_predictions': torch.tensor(ml_predictions, dtype=torch.float32),
            'true_labels': torch.tensor(true_labels, dtype=torch.float32),
            'Image_ID': image_id
        }

# 自定义数据集类
class FusionDataset(Dataset):
    def __init__(self, data, feature_file):
        self.data = data.fillna(0)  # 填充数据集中的空值为0
        self.features = np.load(feature_file, allow_pickle=True).item()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_id = row['Image_ID']

        # 加载图像特征
        image_features = self.features.get(image_id, np.zeros(2048))

        expert_labels = row[['Fracture_Expert_Label', 'Pneumothorax_Expert_Label',
                             'Airspace_Opacity_Expert_Label', 'Nodule_Or_Mass_Expert_Label']].values.astype(float)
 
        ml_predictions = row[['Fracture_ML_Label_alexnet', 'Pneumothorax_ML_Label_alexnet',
                        'Airspace_Opacity_ML_Label_alexnet', 'Nodule_Or_Mass_ML_Label_alexnet']].values.astype(float)  
        
        true_labels = row[['Fracture_GT_Label', 'Pneumothorax_GT_Label',
                           'Airspace_Opacity_GT_Label', 'Nodule_Or_Mass_GT_Label']].values.astype(float)
        
        # print(f"image_features.shape: {image_features.shape}")
        # print(f"expert_labels.shape: {expert_labels.shape}")
        # print(f"ml_predictions.shape: {ml_predictions.shape}")

        return {
            'image_features': torch.tensor(image_features, dtype=torch.float32),
            'expert_labels': torch.tensor(expert_labels, dtype=torch.float32),
            'ml_predictions': torch.tensor(ml_predictions, dtype=torch.float32),
            'true_labels': torch.tensor(true_labels, dtype=torch.float32),
            'Image_ID': image_id
        }


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, num_heads=2, num_layers=2, hidden_dim=128):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.transformer(x)

class Gating(nn.Module):
    def __init__(self, input_dim, num_experts, num_labels, dropout_rate=0.4, confusion_matrix=None):
        super(Gating, self).__init__()
        self.num_labels = num_labels
        self.num_experts = num_experts
        # 输入维度调整为 image_features + expert_labels + ml_predictions
        total_input_dim = input_dim + 2 * num_labels  
        self.gates = nn.ModuleList([self._create_gate(total_input_dim, num_experts, dropout_rate) for _ in range(num_labels)])
        
        if confusion_matrix is not None:
            self.register_buffer("confusion_matrix", torch.tensor(confusion_matrix, dtype=torch.float32))  # (num_labels, num_labels)
            self.transformer = TransformerEncoder(input_dim=num_labels)  # Transformer 学习混淆矩阵模式
            self.confusion_proj = nn.Linear(num_labels, num_experts)  # 让 transformer 输出的信息影响 gating

    def _create_gate(self, input_dim, num_experts, dropout_rate):
        gate = nn.Sequential(
            nn.Linear(input_dim, 256),  # 增加第一层宽度
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.LayerNorm(128),         # 改用LayerNorm
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate/2),# 降低dropout率
            nn.Linear(128, num_experts),
            nn.Softmax(dim=1)
        )
        # 添加参数初始化
        for layer in gate:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='leaky_relu')
                nn.init.constant_(layer.bias, 0.1)
        return gate

    def forward(self, x):
        """
        x: (batch_size, input_dim)
        返回: (batch_size, num_labels, num_experts)
        """
        gating_weights = [gate(x) for gate in self.gates]  # (batch_size, num_labels, num_experts)
        gating_weights = torch.stack(gating_weights, dim=1)  # (batch_size, num_labels, num_experts)

        # 处理混淆矩阵信息
        if hasattr(self, "confusion_matrix"):
            # 通过 Transformer 学习混淆矩阵的特征
            learned_confusion = self.transformer(self.confusion_matrix.unsqueeze(0)).squeeze(0)  # (num_labels, num_labels)
            
            # 投影到 gating 维度，使 gating 计算时受影响
            confusion_gating_factors = self.confusion_proj(learned_confusion)  # (num_labels, num_experts)

            # 让所有 batch 共享相同的 confusion_gating_factors
            confusion_gating_factors = confusion_gating_factors.unsqueeze(0).expand(x.shape[0], -1, -1)  # (batch_size, num_labels, num_experts)

            # 通过 softmax 重新计算 gating 权重，使 gating 受混淆信息影响
            gating_weights = torch.softmax(gating_weights * torch.sigmoid(confusion_gating_factors), dim=2)

        return gating_weights

class MoE(nn.Module):
    def __init__(self, input_dim, num_experts=2, num_labels=4, confusion_matrix=None):
        super(MoE, self).__init__()
        self.gating = Gating(input_dim=input_dim, num_experts=num_experts, num_labels=num_labels, confusion_matrix=confusion_matrix)
    
    def forward(self, image_features, expert_labels,ml_predictions):
        # 拼接三者作为门控输入
        gate_input = torch.cat([image_features, expert_labels, ml_predictions], dim=1)
        gating_weights = self.gating(gate_input)  # 只传递拼接后的输入
        
        # 1. 让 Transformer 学到的混淆信息作用于专家输出
        confusion_matrix = self.gating.confusion_matrix  # (num_labels, num_labels)
        learned_confusion = self.gating.transformer(confusion_matrix.unsqueeze(0)).squeeze(0)  # (num_labels, num_labels)

        # 2. 计算注意力权重，使 human_expert 受影响
        attention_weights = torch.softmax(learned_confusion, dim=1)  # (num_labels, num_labels)

        # 3. 重新调整 expert_labels，使其受到 attention_weights 的影响
        adjusted_expert_labels = torch.matmul(expert_labels, attention_weights)  # (batch_size, num_labels)

        # 4. 组合专家输出 (第一个是人类专家, 第二个是机器专家)
        expert_outputs = torch.stack([adjusted_expert_labels, ml_predictions], dim=2)  # (batch_size, num_labels, num_experts)

        # 5. 计算加权和
        fused_output = torch.sum(expert_outputs * gating_weights, dim=2)  # (batch_size, num_labels)

        return torch.sigmoid(fused_output)



    
# 寻找最优阈值
def find_best_threshold(predictions, true_labels):
    best_thresholds = []
    for i in range(predictions.shape[1]):  # 假设每一列代表一个类别
        thresholds = np.linspace(0, 1, 100)
        losses = [hamming_loss(true_labels[:, i], (predictions[:, i] > t).astype(int)) for t in thresholds]
        best_threshold = thresholds[np.argmin(losses)]
        best_thresholds.append(best_threshold)
    return best_thresholds


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10, model_save_path="best_model.pth"):
    best_val_loss = float('inf')  # 初始值为正无穷
    best_epoch = -1
    # best_thresholds = None  # 用于存储最佳阈值
    best_thresholds = 0.5
    early_stop_counter = 0  # 早停计数器

    # 创建学习率调度器
    # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)  # 这里选择按验证损失调整学习率

    # 记录每个 epoch 的 Hamming Loss 和 AUC
    train_losses = []
    val_losses = []
    val_hamming_losses = []
    val_aucs = []

    for epoch in range(epochs):
        print(f"开始训练 Epoch {epoch + 1}/{epochs}")
        model.train()
        total_train_loss = 0
        
        # 训练阶段
        for batch in train_loader:
            image_features = batch['image_features']
            expert_labels = batch['expert_labels']
            ml_predictions = batch['ml_predictions']
            true_labels = batch['true_labels']
            
            optimizer.zero_grad()
            outputs = model(image_features, expert_labels, ml_predictions)
            loss = criterion(outputs, true_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}")

        # 验证阶段
        model.eval()
        total_val_loss = 0
        all_val_outputs = []
        all_val_labels = []
        all_gating_weights = []  # 用于存储权重
        
        with torch.no_grad():
            for batch in val_loader:
                image_features = batch['image_features']
                expert_labels = batch['expert_labels']
                ml_predictions = batch['ml_predictions']
                true_labels = batch['true_labels']

                outputs = model(image_features, expert_labels, ml_predictions)
                loss = criterion(outputs, true_labels)
                total_val_loss += loss.item()
                all_val_outputs.append(outputs)
                all_val_labels.append(true_labels)
                gate_input = torch.cat([image_features, expert_labels, ml_predictions], dim=1)
                all_gating_weights.append(model.gating(gate_input))  # 获取权重

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Val Loss: {avg_val_loss:.4f}")

        # 计算 Hamming Loss 和 AUC
        all_val_outputs = torch.cat(all_val_outputs).numpy()
        all_val_labels = torch.cat(all_val_labels).numpy()
        val_hamming_loss = hamming_loss(all_val_labels, (all_val_outputs >= 0.5).astype(int))
        val_auc = roc_auc_score(all_val_labels, all_val_outputs, average='macro')
        val_hamming_losses.append(val_hamming_loss)
        val_aucs.append(val_auc)
        print(f"Epoch {epoch + 1}/{epochs}, Val Hamming Loss: {val_hamming_loss:.4f}, Val AUC: {val_auc:.4f}")

        # 输出权重
        all_gating_weights = torch.cat(all_gating_weights).numpy()
        print(f"Epoch {epoch + 1}/{epochs}, Gating Weights: {all_gating_weights}")

        # 检查是否是当前最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            early_stop_counter = 0  # 重新计数
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }, model_save_path)
            print(f"新最佳模型保存于 Epoch {epoch + 1}, Val Loss: {avg_val_loss:.4f}")

            # 计算最佳阈值
            # best_thresholds = find_best_threshold(all_val_outputs, all_val_labels)
        else:
            early_stop_counter += 1

        if early_stop_counter >= 3:
            print(f"🔥 训练提前停止于 Epoch {epoch+1}，因为验证损失 3 轮没有下降")
            break
        # 更新学习率
        # scheduler.step(avg_val_loss)  # 每个 epoch 结束后根据验证损失更新学习率

    print(f"训练完成，最佳模型在 Epoch {best_epoch}，验证损失为 {best_val_loss:.4f}")
    
    return best_thresholds

# # 测试模型
from sklearn.metrics import average_precision_score

def calculate_map(predictions, true_labels):
    """
    计算 Mean Average Precision (MAP)。
    
    :param predictions: 模型预测的概率值，shape (batch_size, num_labels)，NumPy 数组格式
    :param true_labels: 真实标签，shape (batch_size, num_labels)，NumPy 数组格式
    :return: MAP 值
    """
    num_labels = true_labels.shape[1]  # 使用 NumPy 数组的 shape 属性
    aps = []

    for i in range(num_labels):
        ap = average_precision_score(true_labels[:, i], predictions[:, i])
        aps.append(ap)

    map_value = np.mean(aps)  # 计算 MAP
    return map_value
from sklearn.metrics import roc_auc_score

def evaluate_model(model, loader, thresholds, model_path='best_model.pth', output_csv='fusion_results.csv'):
    # 加载保存的最优模型
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"已加载最佳模型，Epoch {checkpoint['epoch']}，验证损失为 {checkpoint['loss']:.4f}")

    all_outputs = []
    all_labels = []
    all_image_ids = []
    all_expert_labels = []
    all_ml_predictions = []
    all_gating_weights = []  # 用于存储权重

    with torch.no_grad():
        for batch in loader:
            image_features = batch['image_features']
            expert_labels = batch['expert_labels']
            ml_predictions = batch['ml_predictions']
            true_labels = batch['true_labels']
            image_ids = batch['Image_ID']

            outputs = model(image_features, expert_labels, ml_predictions)
            all_outputs.append(outputs)
            all_labels.append(true_labels)
            all_image_ids.extend(image_ids)
            all_expert_labels.append(expert_labels)
            all_ml_predictions.append(ml_predictions)
            gate_input = torch.cat([image_features, expert_labels, ml_predictions], dim=1)
            all_gating_weights.append(model.gating(gate_input))  # 获取权重

    all_outputs = torch.cat(all_outputs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_expert_labels = torch.cat(all_expert_labels).numpy()
    all_ml_predictions = torch.cat(all_ml_predictions).numpy()
    all_gating_weights = torch.cat(all_gating_weights).numpy()

    # 应用阈值
    predictions = (all_outputs >= np.array(thresholds)).astype(int)
    fusion_hamming_loss = hamming_loss(all_labels, predictions)
    expert_hamming_loss = hamming_loss(all_labels, (all_expert_labels >= 0.5).astype(int))
    ml_hamming_loss = hamming_loss(all_labels, (all_ml_predictions >= 0.5).astype(int))



    # print(f"融合后Hamming Loss: {fusion_hamming_loss:.4f}")

    # 计算Accuracy
    accuracy_fusion = (predictions == all_labels).mean()
    print(f"融合后Accuracy: {accuracy_fusion:.4f}")

    map_fusion = calculate_map(all_outputs, all_labels)
    map_expert = calculate_map(all_expert_labels, all_labels)
    map_ml = calculate_map(all_ml_predictions, all_labels)


    # 计算AUC
    auc_fusion = roc_auc_score(all_labels, all_outputs, average='macro')
    auc_expert = roc_auc_score(all_labels, all_expert_labels, average='macro')
    auc_ml = roc_auc_score(all_labels, all_ml_predictions, average='macro')

    # 输出权重
    # print(f"Gating Weights: {all_gating_weights}")

    f1_fusion = f1_score(all_labels, predictions, average='macro')
    f1_expert = f1_score(all_labels, (all_expert_labels >= 0.5).astype(int), average='macro')
    f1_ml = f1_score(all_labels, (all_ml_predictions >= 0.5).astype(int), average='macro')

    precision_fusion = precision_score(all_labels, predictions, average='macro')
    precision_expert = precision_score(all_labels, (all_expert_labels >= 0.5).astype(int), average='macro')
    precision_ml = precision_score(all_labels, (all_ml_predictions >= 0.5).astype(int), average='macro')

    recall_fusion = recall_score(all_labels, predictions, average='macro')
    recall_expert = recall_score(all_labels, (all_expert_labels >= 0.5).astype(int), average='macro')
    recall_ml = recall_score(all_labels, (all_ml_predictions >= 0.5).astype(int), average='macro')


    print(f"融合结果 Hamming_loss:{fusion_hamming_loss:.4f}, MAP:{map_fusion:.4f}, AUC: {auc_fusion:.4f}, F1: {f1_fusion:.4f}, Precision: {precision_fusion:.4f}, Recall: {recall_fusion:.4f}")
    print(f"人类专家 Hamming_loss:{expert_hamming_loss:.4f}, MAP:{map_expert:.4f}, AUC: {auc_expert:.4f}, F1: {f1_expert:.4f}, Precision: {precision_expert:.4f}, Recall: {recall_expert:.4f}")
    print(f"机器专家 Hamming_loss:{ml_hamming_loss:.4f}, MAP:{map_ml:.4f}, AUC: {auc_ml:.4f}, F1: {f1_ml:.4f}, Precision: {precision_ml:.4f}, Recall: {recall_ml:.4f}")

    # print(f"Gating Weights: {all_gating_weights}")
    # 计算每个标签的AUC
    auc_fusion_per_label = []
    for i, label in enumerate(['Fracture', 'Pneumothorax', 'Airspace_Opacity', 'Nodule_Or_Mass']):
        try:
            auc = roc_auc_score(all_labels[:, i], all_outputs[:, i])
            auc_fusion_per_label.append(auc)
            print(f"{label} AUC: {auc:.4f}")
        except ValueError:
            auc_fusion_per_label.append(np.nan)
            print(f"{label} AUC: 无法计算（可能所有样本属于同一类别）")

    results_df = pd.DataFrame(all_outputs, columns=['Fracture', 'Pneumothorax', 'Airspace_Opacity', 'Nodule_Or_Mass'])
    results_df['Image_ID'] = all_image_ids
    results_df.to_csv(output_csv, index=False)
    print(f"结果已保存到 {output_csv}")

    log_file='test0420_evaluation_wo_thr.txt'
    with open(log_file, 'a') as f:
        f.write(f"融合后Hamming Loss: {fusion_hamming_loss:.4f}\n")
        f.write(f"融合结果 MAP:{map_fusion:.4f}, AUC: {auc_fusion:.4f}, F1: {f1_fusion:.4f}, Precision: {precision_fusion:.4f}, Recall: {recall_fusion:.4f}\n")
        f.write(f"人类专家 MAP:{map_expert:.4f}, AUC: {auc_expert:.4f}, F1: {f1_expert:.4f}, Precision: {precision_expert:.4f}, Recall: {recall_expert:.4f}\n")
        f.write(f"机器专家 MAP:{map_ml:.4f}, AUC: {auc_ml:.4f}, F1: {f1_ml:.4f}, Precision: {precision_ml:.4f}, Recall: {recall_ml:.4f}\n")

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

confusion_matrix = np.array([
    [0.42, 0.01, 0.23, 0.02],
    [0.01, 0.71, 0.19, 0.02],
    [0.00, 0.01, 0.83, 0.03],
    [0.01, 0.01, 0.17, 0.62]
])



# 设置路径
image_folder = '/data/user24262904/My/ChestX-ray14/images'
data_path = "all_predictions.csv"

data = pd.read_csv(data_path)


# 提取并保存特征
feature_file = 'image_features.npy'
if not os.path.exists(feature_file):
    extract_and_save_features(data, image_folder, transform, feature_file)


# 数据分割
train_data = data[data['Set_ID_x'] == 'train']
val_data = data[data['Set_ID_x'] == 'val']
test_data = data[data['Set_ID_x'] == 'test']

train_dataset = FusionDataset(train_data, feature_file)
val_dataset = FusionDataset(val_data, feature_file)
test_dataset = FusionDataset(test_data, feature_file)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        
# 初始化模型
# 初始化MoE模型
print("训练第1次-----------------------------------")
# 运行该函数，确保每次训练都是相同的初始条件
model1 = MoE(input_dim=2048, num_experts=2, num_labels=4, confusion_matrix=confusion_matrix)  # 输入维度为图像特征的维度，标签数量为4
criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer1 = optim.Adam(model1.parameters(), lr=0.00001)  # 调低学习率,增大梯度就变为0了
# 训练和测试
thresholds1 = train_model(model1, train_loader, val_loader, criterion, optimizer1, epochs=30, model_save_path='best_model1.pth')
evaluate_model(model1, test_loader, thresholds=thresholds1, model_path='best_model1.pth', output_csv='test_results1.csv')

# # 数据分割
# train_data = data[data['Set_ID_x'] == 'train']
# val_data = data[data['Set_ID_x'] == 'val']
# test_data = data[data['Set_ID_x'] == 'test']

# train_dataset = FusionDataset(train_data, feature_file)
# val_dataset = FusionDataset(val_data, feature_file)
# test_dataset = FusionDataset(test_data, feature_file)

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        
# # 初始化模型
# # 初始化MoE模型
# print("训练第2次-----------------------------------")
# # 运行该函数，确保每次训练都是相同的初始条件
# model1 = MoE(input_dim=2048, num_experts=2, num_labels=4, confusion_matrix=confusion_matrix)  # 输入维度为图像特征的维度，标签数量为4
# criterion = nn.BCELoss()  # 二分类交叉熵损失
# optimizer1 = optim.Adam(model1.parameters(), lr=0.00001)  # 调低学习率,增大梯度就变为0了
# # 训练和测试
# thresholds1 = train_model(model1, train_loader, val_loader, criterion, optimizer1, epochs=30, model_save_path='best_model1.pth')
# evaluate_model(model1, test_loader, thresholds=thresholds1, model_path='best_model1.pth', output_csv='test_results1.csv')

# # 数据分割
# train_data = data[data['Set_ID_x'] == 'train']
# val_data = data[data['Set_ID_x'] == 'val']
# test_data = data[data['Set_ID_x'] == 'test']

# train_dataset = FusionDataset(train_data, feature_file)
# val_dataset = FusionDataset(val_data, feature_file)
# test_dataset = FusionDataset(test_data, feature_file)

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        
# # 初始化模型
# # 初始化MoE模型
# print("训练第3次-----------------------------------")
# # 运行该函数，确保每次训练都是相同的初始条件
# model1 = MoE(input_dim=2048, num_experts=2, num_labels=4, confusion_matrix=confusion_matrix)  # 输入维度为图像特征的维度，标签数量为4
# criterion = nn.BCELoss()  # 二分类交叉熵损失
# optimizer1 = optim.Adam(model1.parameters(), lr=0.00001)  # 调低学习率,增大梯度就变为0了
# # 训练和测试
# thresholds1 = train_model(model1, train_loader, val_loader, criterion, optimizer1, epochs=30, model_save_path='best_model1.pth')
# evaluate_model(model1, test_loader, thresholds=thresholds1, model_path='best_model1.pth', output_csv='test_results1.csv')


# # 数据分割
# train_data = data[data['Set_ID_x'] == 'train']
# val_data = data[data['Set_ID_x'] == 'val']
# test_data = data[data['Set_ID_x'] == 'test']

# train_dataset = FusionDataset(train_data, feature_file)
# val_dataset = FusionDataset(val_data, feature_file)
# test_dataset = FusionDataset(test_data, feature_file)

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        
# # 初始化模型
# # 初始化MoE模型
# print("训练第4次-----------------------------------")
# # 运行该函数，确保每次训练都是相同的初始条件
# model1 = MoE(input_dim=2048, num_experts=2, num_labels=4, confusion_matrix=confusion_matrix)  # 输入维度为图像特征的维度，标签数量为4
# criterion = nn.BCELoss()  # 二分类交叉熵损失
# optimizer1 = optim.Adam(model1.parameters(), lr=0.00001)  # 调低学习率,增大梯度就变为0了
# # 训练和测试
# thresholds1 = train_model(model1, train_loader, val_loader, criterion, optimizer1, epochs=30, model_save_path='best_model1.pth')
# evaluate_model(model1, test_loader, thresholds=thresholds1, model_path='best_model1.pth', output_csv='test_results1.csv')


# # 数据分割
# train_data = data[data['Set_ID_x'] == 'train']
# val_data = data[data['Set_ID_x'] == 'val']
# test_data = data[data['Set_ID_x'] == 'test']

# train_dataset = FusionDataset(train_data, feature_file)
# val_dataset = FusionDataset(val_data, feature_file)
# test_dataset = FusionDataset(test_data, feature_file)

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        
# # 初始化模型
# # 初始化MoE模型
# print("训练第5次-----------------------------------")
# # 运行该函数，确保每次训练都是相同的初始条件
# model1 = MoE(input_dim=2048, num_experts=2, num_labels=4, confusion_matrix=confusion_matrix)  # 输入维度为图像特征的维度，标签数量为4
# criterion = nn.BCELoss()  # 二分类交叉熵损失
# optimizer1 = optim.Adam(model1.parameters(), lr=0.00001)  # 调低学习率,增大梯度就变为0了
# # 训练和测试
# thresholds1 = train_model(model1, train_loader, val_loader, criterion, optimizer1, epochs=30, model_save_path='best_model1.pth')
# evaluate_model(model1, test_loader, thresholds=thresholds1, model_path='best_model1.pth', output_csv='test_results1.csv')


# # 数据分割
# train_data = data[data['Set_ID_x'] == 'train']
# val_data = data[data['Set_ID_x'] == 'val']
# test_data = data[data['Set_ID_x'] == 'test']

# train_dataset = FusionDataset(train_data, feature_file)
# val_dataset = FusionDataset(val_data, feature_file)
# test_dataset = FusionDataset(test_data, feature_file)

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        
# # 初始化模型
# # 初始化MoE模型
# print("训练第6次-----------------------------------")
# # 运行该函数，确保每次训练都是相同的初始条件
# model1 = MoE(input_dim=2048, num_experts=2, num_labels=4, confusion_matrix=confusion_matrix)  # 输入维度为图像特征的维度，标签数量为4
# criterion = nn.BCELoss()  # 二分类交叉熵损失
# optimizer1 = optim.Adam(model1.parameters(), lr=0.00001)  # 调低学习率,增大梯度就变为0了
# # 训练和测试
# thresholds1 = train_model(model1, train_loader, val_loader, criterion, optimizer1, epochs=30, model_save_path='best_model1.pth')
# evaluate_model(model1, test_loader, thresholds=thresholds1, model_path='best_model1.pth', output_csv='test_results1.csv')


# # 数据分割
# train_data = data[data['Set_ID_x'] == 'train']
# val_data = data[data['Set_ID_x'] == 'val']
# test_data = data[data['Set_ID_x'] == 'test']

# train_dataset = FusionDataset(train_data, feature_file)
# val_dataset = FusionDataset(val_data, feature_file)
# test_dataset = FusionDataset(test_data, feature_file)

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        
# # 初始化模型
# # 初始化MoE模型
# print("训练第7次-----------------------------------")
# # 运行该函数，确保每次训练都是相同的初始条件
# model1 = MoE(input_dim=2048, num_experts=2, num_labels=4, confusion_matrix=confusion_matrix)  # 输入维度为图像特征的维度，标签数量为4
# criterion = nn.BCELoss()  # 二分类交叉熵损失
# optimizer1 = optim.Adam(model1.parameters(), lr=0.00001)  # 调低学习率,增大梯度就变为0了
# # 训练和测试
# thresholds1 = train_model(model1, train_loader, val_loader, criterion, optimizer1, epochs=30, model_save_path='best_model1.pth')
# evaluate_model(model1, test_loader, thresholds=thresholds1, model_path='best_model1.pth', output_csv='test_results1.csv')


# # 数据分割
# train_data = data[data['Set_ID_x'] == 'train']
# val_data = data[data['Set_ID_x'] == 'val']
# test_data = data[data['Set_ID_x'] == 'test']

# train_dataset = FusionDataset(train_data, feature_file)
# val_dataset = FusionDataset(val_data, feature_file)
# test_dataset = FusionDataset(test_data, feature_file)

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        
# # 初始化模型
# # 初始化MoE模型
# print("训练第8次-----------------------------------")
# # 运行该函数，确保每次训练都是相同的初始条件
# model1 = MoE(input_dim=2048, num_experts=2, num_labels=4, confusion_matrix=confusion_matrix)  # 输入维度为图像特征的维度，标签数量为4
# criterion = nn.BCELoss()  # 二分类交叉熵损失
# optimizer1 = optim.Adam(model1.parameters(), lr=0.00001)  # 调低学习率,增大梯度就变为0了
# # 训练和测试
# thresholds1 = train_model(model1, train_loader, val_loader, criterion, optimizer1, epochs=30, model_save_path='best_model1.pth')
# evaluate_model(model1, test_loader, thresholds=thresholds1, model_path='best_model1.pth', output_csv='test_results1.csv')


# # 数据分割
# train_data = data[data['Set_ID_x'] == 'train']
# val_data = data[data['Set_ID_x'] == 'val']
# test_data = data[data['Set_ID_x'] == 'test']

# train_dataset = FusionDataset(train_data, feature_file)
# val_dataset = FusionDataset(val_data, feature_file)
# test_dataset = FusionDataset(test_data, feature_file)

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        
# # 初始化模型
# # 初始化MoE模型
# print("训练第9次-----------------------------------")
# # 运行该函数，确保每次训练都是相同的初始条件
# model1 = MoE(input_dim=2048, num_experts=2, num_labels=4, confusion_matrix=confusion_matrix)  # 输入维度为图像特征的维度，标签数量为4
# criterion = nn.BCELoss()  # 二分类交叉熵损失
# optimizer1 = optim.Adam(model1.parameters(), lr=0.00001)  # 调低学习率,增大梯度就变为0了
# # 训练和测试
# thresholds1 = train_model(model1, train_loader, val_loader, criterion, optimizer1, epochs=30, model_save_path='best_model1.pth')
# evaluate_model(model1, test_loader, thresholds=thresholds1, model_path='best_model1.pth', output_csv='test_results1.csv')


# # 数据分割
# train_data = data[data['Set_ID_x'] == 'train']
# val_data = data[data['Set_ID_x'] == 'val']
# test_data = data[data['Set_ID_x'] == 'test']

# train_dataset = FusionDataset(train_data, feature_file)
# val_dataset = FusionDataset(val_data, feature_file)
# test_dataset = FusionDataset(test_data, feature_file)

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        
# # 初始化模型
# # 初始化MoE模型
# print("训练第10次-----------------------------------")
# # 运行该函数，确保每次训练都是相同的初始条件
# model1 = MoE(input_dim=2048, num_experts=2, num_labels=4, confusion_matrix=confusion_matrix)  # 输入维度为图像特征的维度，标签数量为4
# criterion = nn.BCELoss()  # 二分类交叉熵损失
# optimizer1 = optim.Adam(model1.parameters(), lr=0.00001)  # 调低学习率,增大梯度就变为0了
# # 训练和测试
# thresholds1 = train_model(model1, train_loader, val_loader, criterion, optimizer1, epochs=30, model_save_path='best_model1.pth')
# evaluate_model(model1, test_loader, thresholds=thresholds1, model_path='best_model1.pth', output_csv='test_results1.csv')
