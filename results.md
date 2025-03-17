# TASK 1
```
Best hyperparameters found:
{'embedding_dim': 200, 'hidden_dim': 100, 'lr': 0.01, 'dropout_rate': 0.3, 'weight_decay': 0.0001}
Best validation accuracy: 30.28%
Test Loss: 1.767 | Test Acc: 16.10%
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
  Validation Loss: 0.473 | Validation Acc: 84.64%
  Test Loss: 0.401 | Test Acc: 88.51%
Architecture: bidirectional_lstm
  Validation Loss: 0.463 | Validation Acc: 85.42%
  Test Loss: 0.428 | Test Acc: 87.48%
Architecture: cnn
  Validation Loss: 0.315 | Validation Acc: 87.67%
  Test Loss: 0.254 | Test Acc: 92.01%
```