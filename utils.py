import time
import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        text, text_lengths = batch.text
        batch_size = text.shape[0]  # 当前批次的实际样本数
        labels = batch.label[:batch_size]  # 截取 labels，使其长度与 batch_size 匹配

        predictions = model(text, text_lengths)
        loss = criterion(predictions, labels)
        loss.backward()
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