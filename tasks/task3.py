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

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, sentence_embedding_method="last", use_packing=True, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sentence_embedding_method = sentence_embedding_method
        self.use_packing = use_packing
        self.dropout = nn.Dropout(dropout)  

        if self.sentence_embedding_method == "attention":
            self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, text, text_lengths):
        vocab_size = self.embedding.num_embeddings
        if text.max().item() >= vocab_size:
            raise ValueError(f"Text contains indices out of vocab range: max index {text.max().item()} >= vocab size {vocab_size}")

        if (text_lengths <= 0).any():
            raise ValueError(f"text_lengths contains non-positive values: {text_lengths}")
        if (text_lengths > text.shape[1]).any():
            raise ValueError(f"text_lengths contains values larger than sequence length: {text_lengths}, max seq length: {text.shape[1]}")

        embedded = self.embedding(text)

        if self.use_packing:
            try:
                text_lengths_cpu = text_lengths.cpu().to(torch.int64)
                if torch.cuda.is_available() and text.device.type == 'cuda':
                    torch.cuda.synchronize()
                packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths_cpu, batch_first=True, enforce_sorted=False)
                packed_output, hidden = self.rnn(packed_embedded)
                output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
            except Exception as e:
                print("Error in pack_padded_sequence:", e)
                raise
        else:
            output, hidden = self.rnn(embedded)

        if self.sentence_embedding_method == "last":
            sentence_embedding = hidden.squeeze(0)
        elif self.sentence_embedding_method == "mean":
            sentence_embedding = torch.mean(output, dim=1)
        elif self.sentence_embedding_method == "max":
            sentence_embedding = torch.max(output, dim=1)[0]
        elif self.sentence_embedding_method == "attention":
            attn_weights = self.attention(output)
            attn_weights = torch.softmax(attn_weights, dim=1)
            sentence_embedding = torch.sum(output * attn_weights, dim=1)
        else:
            raise ValueError("Unknown sentence embedding method!")

        sentence_embedding = self.dropout(sentence_embedding)
        return self.fc(sentence_embedding)

# Define utility functions
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

# For tokenization
TEXT = data.Field(tokenize='spacy',
                  tokenizer_language='en_core_web_sm',
                  include_lengths=True,
                  pad_first=False,  # Ensure padding doesn't interfere with length calculation
                  batch_first=True)

# For multi-class classification labels
LABEL = data.LabelField(dtype=torch.long)

# Load the TREC dataset
train_data, test_data = datasets.TREC.splits(TEXT, LABEL, fine_grained=False)

print(f'Number of training examples: {len(train_data)}')
print(f'Number of testing examples: {len(test_data)}')
print(vars(train_data.examples[0]))

train_data, valid_data = train_data.split(split_ratio=0.8, random_state=random.seed(SEED))

print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')

TEXT.build_vocab(train_data, max_size=10000)
LABEL.build_vocab(train_data)

print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")

# Load pre-trained Word2Vec embeddings
word2vec_vectors = api.load('word2vec-google-news-300')
embedding_dim = 300
vocab_size = len(TEXT.vocab)
embedding_matrix = np.random.uniform(-0.25, 0.25, (vocab_size, embedding_dim))

for word, idx in TEXT.vocab.stoi.items():
    if word in word2vec_vectors:
        embedding_matrix[idx] = word2vec_vectors[word]

embedding_matrix = torch.FloatTensor(embedding_matrix)

# Temporarily use CPU to debug
device = torch.device('cuda')  # Change back to 'cuda' if resolved
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    sort_within_batch=True,
    sort_key=lambda x: len(x.text),
    device=device
)

# Model parameters
VOCAB_SIZE = len(TEXT.vocab)
EMBEDDING_DIM = 300
HIDDEN_DIM = 50
OUTPUT_DIM = len(LABEL.vocab)

# Define different sentence embedding methods to test
sentence_embedding_methods = ["last", "mean", "max", "attention"]

# Dictionary to store results
results = {}

# Loop over different sentence embedding methods
for method in sentence_embedding_methods:
    print(f"\nTraining with sentence embedding method: {method}\n")

    # Initialize the model (disable packing for debugging)
    model = RNN(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, sentence_embedding_method=method, use_packing=True)

    # Load pre-trained embeddings
    model.embedding.weight.data.copy_(embedding_matrix)
    model.embedding.weight.requires_grad = True

    print(f'The model has {count_parameters(model):,} trainable parameters')

    # Move model to device
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss().to(device)

    best_valid_loss = float('inf')

    # Training loop
    for epoch in range(N_EPOCHS):
        start_time = time.time()

        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'task3-model-{method}.pt')

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

    # Load the best model and evaluate on the test set
    model.load_state_dict(torch.load(f'task3-model-{method}.pt'))
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

    # Store results
    results[method] = {
        "valid_loss": best_valid_loss,
        "valid_acc": valid_acc,
        "test_loss": test_loss,
        "test_acc": test_acc
    }

# Print comparison of all methods
print("\nComparison with Task 2:")
for method, result in results.items():
    print(f"Method: {method}")
    print(f"  Validation Loss: {result['valid_loss']:.3f} | Validation Acc: {result['valid_acc']*100:.2f}%")
    print(f"  Test Loss: {result['test_loss']:.3f} | Test Acc: {result['test_acc']*100:.2f}%")