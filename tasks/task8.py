import torch
import numpy as np
import random
from torchtext import data
from torchtext import datasets
import torch.optim as optim
import torch.nn as nn
import os
import gensim.downloader as api
import time
import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

SEED = 1234
BATCH_SIZE = 64
N_EPOCHS = 25

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# 数据增强函数
def get_synonym(word):
    synsets = wordnet.synsets(word)
    if not synsets:
        return word
    synonyms = [lemma.name() for synset in synsets for lemma in synset.lemmas()]
    return synonyms[0] if synonyms else word

def synonym_replacement(text, n=1):
    words = text.split()
    if len(words) <= 1:
        return text
    indices = random.sample(range(len(words)), min(n, len(words)))
    for i in indices:
        words[i] = get_synonym(words[i])
    return ' '.join(words)

def random_deletion(text, p=0.2):
    words = text.split()
    if len(words) <= 1:
        return text
    new_words = [word for word in words if random.random() > p]
    return ' '.join(new_words) if new_words else random.choice(words)

def random_swap(text, n=1):
    words = text.split()
    if len(words) <= 1:
        return text
    for _ in range(n):
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
    return ' '.join(words)

def augment_text(text):
    if random.random() < 0.5:
        choice = random.choice([synonym_replacement, random_deletion, random_swap])
        return choice(text)
    return text

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))
        stdv = 1. / (hidden_dim ** 0.5)
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, x):
        batch_size = x.size(0)
        energy = torch.tanh(self.attn(x.transpose(1, 2)))
        v = self.v.repeat(batch_size, 1).unsqueeze(-1)
        attention = torch.bmm(energy, v).squeeze(-1)
        return torch.softmax(attention, dim=1)

class EnhancedTextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3, padding=1, dilation=1),
            nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
            nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3, padding=4, dilation=4),
            nn.Conv1d(embedding_dim, hidden_dim, kernel_size=5, padding=2, dilation=1)
        ])
        
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.word_attention = Attention(hidden_dim)
        self.sentence_attention = Attention(hidden_dim)
        
        self.residual_proj = nn.Linear(embedding_dim, hidden_dim)
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        embedded = self.dropout(embedded)
        embedded = embedded.transpose(1, 2)
        
        residual = self.residual_proj(embedded.transpose(1, 2)).transpose(1, 2)
        
        conv_outputs = []
        for conv in self.conv_layers:
            conv_out = torch.relu(self.bn(conv(embedded)))
            conv_outputs.append(conv_out)
        
        conv_combined = torch.stack(conv_outputs, dim=0).mean(0) + residual
        
        word_weights = self.word_attention(conv_combined)
        word_context = torch.bmm(conv_combined, word_weights.unsqueeze(-1)).squeeze(-1)
        
        sentence_repr = torch.max(conv_combined, dim=2)[0]
        sentence_weights = self.sentence_attention(conv_combined)
        sentence_context = torch.bmm(conv_combined, sentence_weights.unsqueeze(-1)).squeeze(-1)
        
        final_repr = torch.cat([word_context, sentence_context], dim=1)
        final_repr = self.dropout(final_repr)
        
        return self.fc(final_repr)

TEXT = data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm', 
                 include_lengths=True, pad_first=False, batch_first=True)
LABEL = data.LabelField(dtype=torch.long)

train_data, test_data = datasets.TREC.splits(TEXT, LABEL, fine_grained=False)

# 增强训练数据
from torchtext.data import Example

def augment_dataset(dataset):
    augmented_examples = []
    for ex in dataset.examples:
        orig_text = ' '.join(ex.text)
        aug_text = augment_text(orig_text)
        augmented_examples.append(Example.fromlist([aug_text.split(), ex.label], fields=[('text', TEXT), ('label', LABEL)]))
    return augmented_examples

train_examples = train_data.examples + augment_dataset(train_data)
train_data = data.Dataset(train_examples, fields=[('text', TEXT), ('label', LABEL)])
train_data, valid_data = train_data.split(split_ratio=0.8, random_state=random.seed(SEED))

TEXT.build_vocab(train_data, max_size=20000)
LABEL.build_vocab(train_data)

word2vec_vectors = api.load('word2vec-google-news-300')
embedding_dim = 300
vocab_size = len(TEXT.vocab)
embedding_matrix = np.random.uniform(-0.25, 0.25, (vocab_size, embedding_dim))
for word, idx in TEXT.vocab.stoi.items():
    if word in word2vec_vectors:
        embedding_matrix[idx] = word2vec_vectors[word]
embedding_matrix = torch.FloatTensor(embedding_matrix)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    sort_within_batch=True,
    sort_key=lambda x: len(x.text),
    device=device
)

VOCAB_SIZE = len(TEXT.vocab)
EMBEDDING_DIM = 300
HIDDEN_DIM = 256
OUTPUT_DIM = len(LABEL.vocab)

model = EnhancedTextClassifier(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    output_dim=OUTPUT_DIM,
    dropout=0.3
)

model.embedding.weight.data.copy_(embedding_matrix)
model.embedding.weight.requires_grad = True
model = model.to(device)

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

criterion = LabelSmoothingLoss(classes=OUTPUT_DIM, smoothing=0.1).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.0002, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        text, text_lengths = batch.text
        predictions = model(text, text_lengths)
        loss = criterion(predictions, batch.label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += (predictions.argmax(1) == batch.label).float().mean().item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths)
            loss = criterion(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += (predictions.argmax(1) == batch.label).float().mean().item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

best_valid_loss = float('inf')
for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    scheduler.step()
    
    end_time = time.time()
    epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'enhanced-model.pt')

    print(f'Epoch: {epoch+1:02} | Time: {int(epoch_mins)}m {int(epoch_secs)}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\tVal. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}%')

model.load_state_dict(torch.load('enhanced-model.pt'))
valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
test_loss, test_acc = evaluate(model, test_iterator, criterion)

print(f'\nFinal Results:')
print(f'Validation Loss: {valid_loss:.3f} | Validation Accuracy: {valid_acc*100:.2f}%')
print(f'Test Loss: {test_loss:.3f} | Test Accuracy: {test_acc*100:.2f}%')