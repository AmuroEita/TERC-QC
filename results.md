# TASK 1
```
Best hyperparameters found:
{'embedding_dim': 50, 'hidden_dim': 200, 'lr': 0.005, 'dropout_rate': 0.4, 'weight_decay': 0, 'batch_size': 32}
Best validation accuracy: 26.36%
Final Validation Loss: 1.644 | Validation Acc: 26.36%
Final Test Loss: 1.647 | Test Acc: 23.32%
```

# TASK 2
```
Task 1 Results (Best Hyperparameters):
Validation Accuracy: 28.41%
Test Accuracy: 25.97%

Task 2 Results with Pre-trained Embeddings:

Glove Embeddings:
Validation Accuracy: 26.25%
Test Accuracy: 20.49%

Fasttext Embeddings:
Validation Accuracy: 24.23%
Test Accuracy: 25.24%

Word2vec Embeddings:
Validation Accuracy: 21.91%
Test Accuracy: 21.29%

=== Comparison and Analysis ===

Glove vs Task 1:
Validation Accuracy Difference: -2.16%
Test Accuracy Difference: -5.47%

Fasttext vs Task 1:
Validation Accuracy Difference: -4.18%
Test Accuracy Difference: -0.73%

Word2vec vs Task 1:
Validation Accuracy Difference: -6.50%
Test Accuracy Difference: -4.67%
```

# TASK 3
```
Comparison with Task 2:
Method: last
  Validation Loss: 1.633 | Validation Acc: 25.26%
  Test Loss: 1.657 | Test Acc: 32.03%
Method: mean
  Validation Loss: 1.601 | Validation Acc: 41.84%
  Test Loss: 1.617 | Test Acc: 45.84%
Method: max
  Validation Loss: 1.520 | Validation Acc: 44.62%
  Test Loss: 1.527 | Test Acc: 45.00%
Method: attention
  Validation Loss: 1.580 | Validation Acc: 42.62%
  Test Loss: 1.610 | Test Acc: 45.13%
```

# TASK 4
```
Task 4 Results and Comparison with Task 3 Best (max method):
Task 3 Best (max method):
  Validation Loss: 1.520 | Validation Acc: 44.62%
  Test Loss: 1.527 | Test Acc: 45.00%

Task 4 Results:
Architecture: bidirectional_gru
  Validation Loss: 0.500 | Validation Acc: 80.64%
  Test Loss: 0.427 | Test Acc: 88.25%
Architecture: bidirectional_lstm
  Validation Loss: 0.501 | Validation Acc: 85.85%
  Test Loss: 0.461 | Test Acc: 84.65%
Architecture: cnn
  Validation Loss: 0.317 | Validation Acc: 87.93%
  Test Loss: 0.254 | Test Acc: 91.18%
```

# TASK 5
```
CNN and Without Data arguement ?
Validation Loss: 0.708 | Validation Accuracy: 87.93%
Test Loss: 0.659 | Test Accuracy: 90.75%

CNN and Data arguement
Validation Loss: 0.708 | Validation Accuracy: 85.76%
Test Loss: 0.659 | Test Accuracy: 87.82%

With CNN and Without Data arguement
Validation Loss: 0.708 | Validation Accuracy: 89.32%
Test Loss: 0.659 | Test Accuracy: 92.16%

New CNN and Data arguement
Validation Loss: 0.590 | Validation Accuracy: 94.97%
Test Loss: 0.639 | Test Accuracy: 93.43%
```