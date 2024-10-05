import matplotlib.pyplot as plt
import numpy as np
from task2_c import test_values, test_labels, predicted_labels

# Misclassified vs correctly classified data

# Identify indices of correctly and incorrectly classified samples
correct = np.where(test_labels == predicted_labels)[0]
misclassified = np.where(test_labels != predicted_labels)[0]

# Extract data for plotting
correct_values = test_values[correct]
correct_labels = test_labels[correct]

incorrect_values = test_values[misclassified]
incorrect_labels = test_labels[misclassified]

if __name__ == '__main__':
    # Plotting
    plt.scatter(correct_values, correct_labels, facecolors='none', edgecolors='green', label='Correctly classified')
    plt.scatter(incorrect_values, incorrect_labels,facecolors='none', edgecolors='red', label='Misclassified')
    plt.xlabel('Values')
    plt.ylabel('True Labels')
    plt.title('Misclassified and Correctly Classified Test Data')
    plt.legend()
    plt.show()