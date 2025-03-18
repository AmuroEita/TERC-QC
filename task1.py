import torch
import numpy as np
import random
from torchtext import data
from torchtext import datasets
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import os
import time
from random import choice

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, text, text_lengths):
        batch_size = text.shape[0]
        text_lengths = text_lengths[:batch_size]
        max_seq_len = text.shape[1]
        text_lengths = torch.clamp(text_lengths, min=1, max=max_seq_len)
        text_lengths = text_lengths.cpu().long()
        embedded = self.dropout(self.embedding(text))
        packed_embedded = pack_padded_sequence(embedded, text_lengths, batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.rnn(packed_embedded)
        hidden = self.dropout(hidden.squeeze(0))
        return self.fc(hidden)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()

    for batch in iterator:
        optimizer.zero_grad()
        text, text_lengths = batch.text
        batch_size = text.shape[0]
        labels = batch.label[:batch_size]

        predictions = model(text, text_lengths)
        loss = criterion(predictions, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted_classes = predictions.max(dim=1)
        correct_predictions = (predicted_classes == labels).float()
        batch_acc = correct_predictions.mean().item()
        epoch_acc += batch_acc

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            batch_size = text.shape[0]
            labels = batch.label[:batch_size]

            predictions = model(text, text_lengths)
            loss = criterion(predictions, labels)
            epoch_loss += loss.item()

            _, predicted_classes = predictions.max(dim=1)
            correct_predictions = (predicted_classes == labels).float()
            batch_acc = correct_predictions.mean().item()
            epoch_acc += batch_acc

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEXT = data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm', include_lengths=True)
LABEL = data.LabelField()

train_data, test_data = datasets.TREC.splits(TEXT, LABEL, fine_grained=False)
train_data, valid_data = train_data.split(split_ratio=0.8, random_state=random.seed(SEED))

print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')

TEXT.build_vocab(train_data, max_size=10000)
LABEL.build_vocab(train_data)

print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")

param_space = {
    'embedding_dim': [50, 100, 200, 300],
    'hidden_dim': [50, 100, 200],
    'lr': [0.01, 0.005, 0.001, 0.0005],
    'dropout_rate': [0.2, 0.3, 0.4, 0.5],
    'weight_decay': [1e-3, 1e-4, 1e-5, 0],
    'batch_size': [32, 64, 128]
}

N_EPOCHS = 20  
N_EXPERIMENTS = 50  

def run_random_search():
    best_valid_acc = 0
    best_params = None
    best_model_state = None

    for idx in range(N_EXPERIMENTS):
        params = {
            'embedding_dim': choice(param_space['embedding_dim']),
            'hidden_dim': choice(param_space['hidden_dim']),
            'lr': choice(param_space['lr']),
            'dropout_rate': choice(param_space['dropout_rate']),
            'weight_decay': choice(param_space['weight_decay']),
            'batch_size': choice(param_space['batch_size'])
        }

        print(f"\nExperiment {idx+1}/{N_EXPERIMENTS}: {params}")

        train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
            (train_data, valid_data, test_data),
            batch_size=params['batch_size'],
            sort_within_batch=True,
            sort_key=lambda x: len(x.text),
            device=device
        )

        VOCAB_SIZE = len(TEXT.vocab)
        OUTPUT_DIM = len(LABEL.vocab)
        model = RNN(VOCAB_SIZE, params['embedding_dim'], params['hidden_dim'], OUTPUT_DIM, params['dropout_rate']).to(device)
        print(f'The model has {count_parameters(model):,} trainable parameters')

        optimizer = optim.SGD(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        criterion = nn.CrossEntropyLoss().to(device)

        best_valid_loss = float('inf')
        best_epoch_valid_acc = 0

        for epoch in range(N_EPOCHS):
            start_time = time.time()

            train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
            valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch_valid_acc = valid_acc
                best_model_state = model.state_dict()

            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\tVal. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}%')

        if best_epoch_valid_acc > best_valid_acc:
            best_valid_acc = best_epoch_valid_acc
            best_params = params
            torch.save(best_model_state, 'best_model.pt')

    return best_params, best_valid_acc

best_params, best_valid_acc = run_random_search()

print("\nBest hyperparameters found:")
print(best_params)
print(f"Best validation accuracy: {best_valid_acc*100:.2f}%")

VOCAB_SIZE = len(TEXT.vocab)
OUTPUT_DIM = len(LABEL.vocab)
best_model = RNN(
    VOCAB_SIZE,
    best_params['embedding_dim'],
    best_params['hidden_dim'],
    OUTPUT_DIM,
    best_params['dropout_rate']
).to(device)

optimizer = optim.SGD(best_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
criterion = nn.CrossEntropyLoss().to(device)
best_model.load_state_dict(torch.load('best_model.pt'))

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=best_params['batch_size'],
    sort_within_batch=True,
    sort_key=lambda x: len(x.text),
    device=device
)

test_loss, test_acc = evaluate(best_model, test_iterator, criterion)
valid_loss, valid_acc = evaluate(best_model, valid_iterator, criterion)
print(f"\nFinal Validation Loss: {valid_loss:.3f} | Validation Acc: {valid_acc*100:.2f}%")
print(f"Final Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%")
