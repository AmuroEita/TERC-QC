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

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

SEED = 1234
BATCH_SIZE = 64
N_EPOCHS = 10

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class SentenceClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, architecture="cnn", dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.architecture = architecture.lower()
        self.dropout = nn.Dropout(dropout)

        if self.architecture == "cnn":
            self.conv1 = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=2, padding=1)
            self.conv2 = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3, padding=1)
            self.conv3 = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=4, padding=2)
            self.pool = nn.AdaptiveMaxPool1d(1)

        final_hidden_dim = hidden_dim * 3  
        self.fc = nn.Linear(final_hidden_dim, output_dim)

    def forward(self, text, text_lengths):
        vocab_size = self.embedding.num_embeddings
        if text.max().item() >= vocab_size:
            raise ValueError(f"Text contains indices out of vocab range: max index {text.max().item()} >= vocab size {vocab_size}")

        if (text_lengths <= 0).any():
            raise ValueError(f"text_lengths contains non-positive values: {text_lengths}")
        if (text_lengths > text.shape[1]).any():
            raise ValueError(f"text_lengths contains values larger than sequence length: {text_lengths}, max seq length: {text.shape[1]}")

        embedded = self.embedding(text)
        embedded = embedded.transpose(1, 2)  
        conv1_out = torch.relu(self.conv1(embedded))
        conv2_out = torch.relu(self.conv2(embedded))
        conv3_out = torch.relu(self.conv3(embedded))
        pool1 = self.pool(conv1_out).squeeze(-1)
        pool2 = self.pool(conv2_out).squeeze(-1)
        pool3 = self.pool(conv3_out).squeeze(-1)
        sentence_embedding = torch.cat((pool1, pool2, pool3), dim=1)

        sentence_embedding = self.dropout(sentence_embedding)
        return self.fc(sentence_embedding)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()

    for batch in iterator:
        optimizer.zero_grad()
        text, text_lengths = batch.text

        predictions = model(text, text_lengths).squeeze(1)
        loss = criterion(predictions, batch.label)

        _, predicted = torch.max(predictions, 1)
        correct = (predicted == batch.label).float()
        acc = correct.sum() / len(batch.label)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)
            loss = criterion(predictions, batch.label)

            _, predicted = torch.max(predictions, 1)
            correct = (predicted == batch.label).float()
            acc = correct.sum() / len(batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

TEXT = data.Field(tokenize='spacy',
                  tokenizer_language='en_core_web_sm',
                  include_lengths=True,
                  pad_first=False,
                  batch_first=True)

LABEL = data.LabelField(dtype=torch.long)

train_data, test_data = datasets.TREC.splits(TEXT, LABEL, fine_grained=False)

train_data, valid_data = train_data.split(split_ratio=0.8, random_state=random.seed(SEED))

TEXT.build_vocab(train_data, max_size=10000)
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
HIDDEN_DIM = 50
OUTPUT_DIM = len(LABEL.vocab)

model = SentenceClassifier(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    output_dim=OUTPUT_DIM,
    architecture="cnn",
    dropout=0.5
)

model.embedding.weight.data.copy_(embedding_matrix)
model.embedding.weight.requires_grad = True

model = model.to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    start_time = time.time()

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'task4-model-cnn.pt')

model.load_state_dict(torch.load('task4-model-cnn.pt'))
valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
print(f'Validation Accuracy: {valid_acc*100:.2f}%')