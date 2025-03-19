import torch
import numpy as np
import random
from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe, FastText, Vectors
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import os
import time
import gensim.downloader as api

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout_rate=0.5, pretrained_embeddings=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # If pretrained embeddings are provided, load them into the embedding layer
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = True  # Allow fine-tuning of embeddings
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

# Compute epoch time
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

BATCH_SIZE = 32
N_EPOCHS = 10

TEXT = data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm', include_lengths=True)
LABEL = data.LabelField()

train_data, test_data = datasets.TREC.splits(TEXT, LABEL, fine_grained=False)
train_data, valid_data = train_data.split(split_ratio=0.8, random_state=random.seed(SEED))

print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')

best_params = {
    'embedding_dim': 50,
    'hidden_dim': 200,
    'lr': 0.005,
    'dropout_rate': 0.4,
    'weight_decay': 0
}

def build_vocab_with_pretrained(embedding_type):
    if embedding_type == 'glove':
        TEXT.build_vocab(train_data, max_size=10000, vectors=GloVe(name='6B', dim=best_params['embedding_dim']))
        return TEXT.vocab.vectors
    elif embedding_type == 'fasttext':
        TEXT.build_vocab(train_data, max_size=10000, vectors=FastText(language='en'))
        return TEXT.vocab.vectors
    elif embedding_type == 'word2vec':
        # Load Word2Vec using gensim
        word2vec_model = api.load('word2vec-google-news-300')
        TEXT.build_vocab(train_data, max_size=10000)
        # Create an embedding matrix for the vocabulary
        embedding_dim = 300  # Word2Vec has 300 dimensions
        pretrained_embeddings = torch.zeros(len(TEXT.vocab), embedding_dim)
        for i, token in enumerate(TEXT.vocab.itos):
            if token in word2vec_model:
                pretrained_embeddings[i] = torch.tensor(word2vec_model[token])
            else:
                pretrained_embeddings[i] = torch.randn(embedding_dim)  # Random init for OOV
        return pretrained_embeddings
    else:
        TEXT.build_vocab(train_data, max_size=10000)  # No pretrained embeddings (Task 1)
        return None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    sort_within_batch=True,
    sort_key=lambda x: len(x.text),
    device=device
)

def run_task1():
    print("\n=== Running Task 1 with Best Hyperparameters ===")
    pretrained_embeddings = build_vocab_with_pretrained(None)  # No pretrained embeddings

    LABEL.build_vocab(train_data)
    VOCAB_SIZE = len(TEXT.vocab)
    OUTPUT_DIM = len(LABEL.vocab)
    model = RNN(
        VOCAB_SIZE,
        best_params['embedding_dim'],
        best_params['hidden_dim'],
        OUTPUT_DIM,
        best_params['dropout_rate'],
        pretrained_embeddings=pretrained_embeddings
    ).to(device)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.SGD(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    criterion = nn.CrossEntropyLoss().to(device)

    best_valid_loss = float('inf')
    best_valid_acc = 0
    best_model_state = None

    for epoch in range(N_EPOCHS):
        start_time = time.time()

        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_valid_acc = valid_acc
            best_model_state = model.state_dict()

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\tVal. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}%')

    model.load_state_dict(best_model_state)
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    print(f"\nTask 1 Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%")

    return best_valid_acc, test_acc

def run_task2(embedding_type):
    print(f"\n=== Running Task 2 with {embedding_type.capitalize()} Embeddings ===")
    
    # Build vocabulary and load pretrained embeddings
    pretrained_embeddings = build_vocab_with_pretrained(embedding_type)

    LABEL.build_vocab(train_data)
    VOCAB_SIZE = len(TEXT.vocab)
    OUTPUT_DIM = len(LABEL.vocab)
    embedding_dim = best_params['embedding_dim']
    if pretrained_embeddings is not None:
        embedding_dim = pretrained_embeddings.shape[1]

    model = RNN(
        VOCAB_SIZE,
        embedding_dim,
        best_params['hidden_dim'],
        OUTPUT_DIM,
        best_params['dropout_rate'],
        pretrained_embeddings=pretrained_embeddings
    ).to(device)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.SGD(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    criterion = nn.CrossEntropyLoss().to(device)

    best_valid_loss = float('inf')
    best_valid_acc = 0
    best_model_state = None

    for epoch in range(N_EPOCHS):
        start_time = time.time()

        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_valid_acc = valid_acc
            best_model_state = model.state_dict()

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\tVal. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}%')

    # Load the best model and evaluate on the test set
    model.load_state_dict(best_model_state)
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    print(f"\nTask 2 ({embedding_type.capitalize()}) Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%")

    return best_valid_acc, test_acc

task1_valid_acc, task1_test_acc = run_task1()

embedding_types = ['glove', 'fasttext', 'word2vec']
task2_results = {}

for emb_type in embedding_types:
    valid_acc, test_acc = run_task2(emb_type)
    task2_results[emb_type] = {'valid_acc': valid_acc, 'test_acc': test_acc}

print("\n=== Final Results ===")
print(f"\nTask 1 Results (Best Hyperparameters):")
print(f"Validation Accuracy: {task1_valid_acc*100:.2f}%")
print(f"Test Accuracy: {task1_test_acc*100:.2f}%")

print("\nTask 2 Results with Pre-trained Embeddings:")
for emb_type, result in task2_results.items():
    print(f"\n{emb_type.capitalize()} Embeddings:")
    print(f"Validation Accuracy: {result['valid_acc']*100:.2f}%")
    print(f"Test Accuracy: {result['test_acc']*100:.2f}%")

print("\n=== Comparison and Analysis ===")
for emb_type, result in task2_results.items():
    valid_diff = result['valid_acc'] - task1_valid_acc
    test_diff = result['test_acc'] - task1_test_acc
    print(f"\n{emb_type.capitalize()} vs Task 1:")
    print(f"Validation Accuracy Difference: {valid_diff*100:+.2f}%")
    print(f"Test Accuracy Difference: {test_diff*100:+.2f}%")