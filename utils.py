import time
import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, iterator, optimizer, criterion):

    epoch_loss = 0
    total_correct = 0
    total_instances = 0

    model.train()

    for batch in iterator:

        optimizer.zero_grad()

        text, text_lengths = batch.text

        # Assuming your model's forward method automatically handles padding, then no need to pack sequence here
        predictions = model(text, text_lengths)

        loss = criterion(predictions, batch.label)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # Compute the number of correct predictions
        _, predicted_classes = predictions.max(dim=1)
        correct_predictions = (predicted_classes == batch.label).float()  # Convert to float for summation
        total_correct += correct_predictions.sum().item()
        total_instances += batch.label.size(0)

    epoch_acc = total_correct / total_instances

    return epoch_loss / len(iterator), epoch_acc

def evaluate(model, iterator, criterion):

    epoch_loss = 0
    total_correct = 0
    total_instances = 0

    model.eval()

    with torch.no_grad():

        for batch in iterator:
            text, text_lengths = batch.text

            # Assuming your model's forward method automatically handles padding, then no need to pack sequence here
            predictions = model(text, text_lengths)

            loss = criterion(predictions, batch.label)
            epoch_loss += loss.item()

            # Compute the number of correct predictions
            _, predicted_classes = predictions.max(dim=1)
            correct_predictions = (predicted_classes == batch.label).float()  # Convert to float for summation
            total_correct += correct_predictions.sum().item()
            total_instances += batch.label.size(0)

    epoch_acc = total_correct / total_instances
    return epoch_loss / len(iterator), epoch_acc

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
