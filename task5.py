import torch
import numpy as np
import random
from torchtext import data
from torchtext import datasets
import torch.optim as optim
import torch.nn as nn
import os
import time
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import spacy

# 设置环境变量以同步 CUDA 操作
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# 检查设备并设置
print("Checking PyTorch and CUDA compatibility...")
print(f"PyTorch version: {torch.__version__}")
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")
if cuda_available:
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    device = torch.device('cuda')
    try:
        x = torch.rand(3, 3).cuda()
        print("Simple CUDA tensor test successful:", x)
    except RuntimeError as e:
        print(f"CUDA test failed: {e}")
        print("Falling back to CPU.")
        device = torch.device('cpu')
else:
    print("CUDA not available, falling back to CPU.")
    device = torch.device('cpu')
print(f"Using device: {device}")

# 检查 SpaCy 模型
try:
    nlp = spacy.load("en_core_web_sm")
    print("SpaCy model 'en_core_web_sm' loaded successfully!")
    TEXT = data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm', include_lengths=True)
except Exception as e:
    print(f"Error loading SpaCy model: {e}")
    TEXT = data.Field(tokenize=str.split, include_lengths=True)

LABEL = data.LabelField()  # 主任务标签（分类）

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, iterator, optimizer, criterion_class, criterion_reg, alpha=0.5):
    epoch_loss = 0
    epoch_acc = 0
    epoch_reg_loss = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        text, text_lengths = batch.text  # text_lengths 已由 TEXT 字段提供
        batch_size = text.shape[0]
        labels = batch.label[:batch_size].to(device)
        lengths = text_lengths[:batch_size].to(device)  # 直接使用 text_lengths

        class_pred, reg_pred = model(text, text_lengths)
        
        loss_class = criterion_class(class_pred, labels)
        loss_reg = criterion_reg(reg_pred.squeeze(), lengths.float())  # 转换为 float 类型
        loss = alpha * loss_class + (1 - alpha) * loss_reg
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss_class.item()
        epoch_reg_loss += loss_reg.item()
        _, predicted_classes = class_pred.max(dim=1)
        correct_predictions = (predicted_classes == labels).float()
        batch_acc = correct_predictions.mean().item()
        epoch_acc += batch_acc
    return (epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_reg_loss / len(iterator))

def evaluate(model, iterator, criterion_class, criterion_reg, alpha=0.5):
    epoch_loss = 0
    epoch_acc = 0
    epoch_reg_loss = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text  # text_lengths 已由 TEXT 字段提供
            batch_size = text.shape[0]
            labels = batch.label[:batch_size].to(device)
            lengths = text_lengths[:batch_size].to(device)  # 直接使用 text_lengths

            class_pred, reg_pred = model(text, text_lengths)
            
            loss_class = criterion_class(class_pred, labels)
            loss_reg = criterion_reg(reg_pred.squeeze(), lengths.float())  # 转换为 float 类型
            
            epoch_loss += loss_class.item()
            epoch_reg_loss += loss_reg.item()
            _, predicted_classes = class_pred.max(dim=1)
            correct_predictions = (predicted_classes == labels).float()
            batch_acc = correct_predictions.mean().item()
            epoch_acc += batch_acc
    return (epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_reg_loss / len(iterator))

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# 修改后的模型：Attention-based Pooling Model
class MTL_AttentionPooling(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim_class):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc_class = nn.Linear(hidden_dim, output_dim_class)  # 分类头
        self.fc_reg = nn.Linear(hidden_dim, 1)  # 回归头

    def forward(self, text, text_lengths):
        batch_size = text.shape[0]
        text_lengths = text_lengths[:batch_size]
        max_seq_len = text.shape[1]
        text_lengths = torch.clamp(text_lengths, min=1, max=max_seq_len)
        text_lengths = text_lengths.to(torch.long).cpu()
        
        embedded = self.embedding(text)
        packed_embedded = pack_padded_sequence(embedded, text_lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)
        
        attention_weights = self.attention(output)
        attention_weights = torch.softmax(attention_weights, dim=1)
        sentence_embedding = torch.sum(output * attention_weights, dim=1)
        
        class_output = self.fc_class(sentence_embedding)
        reg_output = self.fc_reg(sentence_embedding)
        return class_output, reg_output

# 设置随机种子
SEED = 1234
BATCH_SIZE = 64
N_EPOCHS = 10

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# 加载数据集
train_data, test_data = datasets.TREC.splits(
    text_field=TEXT,
    label_field=LABEL,
    root='.data',
    fine_grained=False
)

# 手动添加 LENGTH 字段（仅用于调试输出，不影响批次）
for example in train_data.examples:
    example.length = len(example.text)
for example in test_data.examples:
    example.length = len(example.text)

print(f'Number of training examples: {len(train_data)}')
print(f'Number of testing examples: {len(test_data)}')
print(vars(train_data.examples[0]))

train_data, valid_data = train_data.split(split_ratio=0.8, random_state=random.seed(SEED))
print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')

# 构建词汇表
TEXT.build_vocab(train_data, max_size=10000)
LABEL.build_vocab(train_data)

print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")

# 创建数据迭代器
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    sort_within_batch=True,
    sort_key=lambda x: len(x.text),
    device=device
)

# 初始化模型
VOCAB_SIZE = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
OUTPUT_DIM_CLASS = len(LABEL.vocab)

model = MTL_AttentionPooling(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM_CLASS)
print(f'The model has {count_parameters(model):,} trainable parameters')

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion_class = nn.CrossEntropyLoss()
criterion_reg = nn.MSELoss()

model = model.to(device)
criterion_class = criterion_class.to(device)
criterion_reg = criterion_reg.to(device)

# 训练循环
best_valid_loss = float('inf')
model_path = 'mtl-attention-model.pt'
alpha = 0.7

for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss, train_acc, train_reg_loss = train(model, train_iterator, optimizer, criterion_class, criterion_reg, alpha)
    valid_loss, valid_acc, valid_reg_loss = evaluate(model, valid_iterator, criterion_class, criterion_reg, alpha)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Class Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Train Reg Loss: {train_reg_loss:.3f}')
    print(f'\tValid Class Loss: {valid_loss:.3f} | Valid Acc: {valid_acc*100:.2f}% | Valid Reg Loss: {valid_reg_loss:.3f}')

# 加载最佳模型并评估
model.load_state_dict(torch.load(model_path))

valid_loss, valid_acc, valid_reg_loss = evaluate(model, valid_iterator, criterion_class, criterion_reg, alpha)
print(f'Final Validation Class Loss: {valid_loss:.3f} | Validation Acc: {valid_acc*100:.2f}% | Validation Reg Loss: {valid_reg_loss:.3f}')

test_loss, test_acc, test_reg_loss = evaluate(model, test_iterator, criterion_class, criterion_reg, alpha)
print(f'Final Test Class Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% | Test Reg Loss: {test_reg_loss:.3f}')