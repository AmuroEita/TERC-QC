import matplotlib.pyplot as plt
import numpy as np

# Data from Task 4
labels = ['Bi-GRU', 'Bi-LSTM', 'CNN']
val_acc = [80.64, 85.85, 87.93]  # Validation accuracy
test_acc = [88.25, 84.65, 91.18]  # Test accuracy

# Bar width and positions
bar_width = 0.35
x = np.arange(len(labels))

# Create figure
plt.figure(figsize=(8, 5))
plt.bar(x - bar_width/2, val_acc, bar_width, label='Validation Accuracy', color='skyblue')
plt.bar(x + bar_width/2, test_acc, bar_width, label='Test Accuracy', color='salmon')

# Add value labels on top of bars
for i in range(len(labels)):
    plt.text(x[i] - bar_width/2, val_acc[i] + 1, f'{val_acc[i]}%', ha='center', va='bottom')
    plt.text(x[i] + bar_width/2, test_acc[i] + 1, f'{test_acc[i]}%', ha='center', va='bottom')

# Customize plot
plt.xlabel('Model Architecture')
plt.ylabel('Accuracy (%)')
plt.xticks(x, labels)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
plt.ylim(0, 100)  # Adjust y-limit for better visibility
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout to leave space for legend
plt.tight_layout()
plt.subplots_adjust(top=0.85)  # Leave more space at the top

# Save and display
plt.savefig('figures/task4.png', dpi=300)
plt.show()