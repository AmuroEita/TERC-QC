import matplotlib.pyplot as plt
import numpy as np

methods = ['Final Token', 'Average Pooling', 'Max Pooling', 'Attention Weighting']
val_acc = [25.26, 41.84, 44.62, 42.62]  
test_acc = [32.03, 45.84, 45.00, 45.13] 

bar_width = 0.35
x = np.arange(len(methods))

plt.figure(figsize=(8, 5))
plt.bar(x - bar_width/2, val_acc, bar_width, label='Validation Accuracy', color='skyblue')
plt.bar(x + bar_width/2, test_acc, bar_width, label='Test Accuracy', color='salmon')

for i in range(len(methods)):
    plt.text(x[i] - bar_width/2, val_acc[i] + 1, f'{val_acc[i]}%', ha='center', va='bottom')
    plt.text(x[i] + bar_width/2, test_acc[i] + 1, f'{test_acc[i]}%', ha='center', va='bottom')

plt.xlabel('Sentence Embedding Method')
plt.ylabel('Accuracy (%)')
plt.xticks(x, methods)
plt.legend()
plt.ylim(0, 60)  
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('../figures/task3.png', dpi=300)
plt.show()