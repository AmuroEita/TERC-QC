import matplotlib.pyplot as plt
import numpy as np

# Labels for the configurations
labels = ['Baseline (CNN + No Aug)', 'CNN + Aug', 'New CNN + No Aug', 'New CNN + Aug']
val_acc = [87.93, 85.76, 89.32, 94.97]  # Validation accuracies
test_acc = [90.75, 87.82, 92.16, 93.43]  # Test accuracies

bar_width = 0.35
x = np.arange(len(labels))

plt.figure(figsize=(8, 5))
plt.bar(x - bar_width/2, val_acc, bar_width, label='Validation Accuracy', color='skyblue')
plt.bar(x + bar_width/2, test_acc, bar_width, label='Test Accuracy', color='salmon')

# Add text labels above bars
for i in range(len(labels)):
    plt.text(x[i] - bar_width/2, val_acc[i] + 1, f'{val_acc[i]}%', ha='center', va='bottom')
    plt.text(x[i] + bar_width/2, test_acc[i] + 1, f'{test_acc[i]}%', ha='center', va='bottom')

plt.xlabel('Model Configuration')
plt.ylabel('Accuracy (%)')
plt.xticks(x, labels)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
plt.ylim(0, 100)  # Adjusted to fit your higher accuracy values
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.subplots_adjust(top=0.85)

# Save the figure (uncomment if needed)
plt.savefig('../figures/task5.png', dpi=300)