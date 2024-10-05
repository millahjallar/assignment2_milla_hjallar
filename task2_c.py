import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from task2_a import zero_values, one_values, data_transposed
from sklearn.model_selection import train_test_split

# Splitting data into training and test sets
zero_train, zero_test = train_test_split(zero_values, test_size=0.3, random_state=42)
one_train, one_test = train_test_split(one_values, test_size=0.3, random_state=42)

test_values = np.concatenate([zero_test, one_test])
test_labels = np.concatenate([np.zeros(len(zero_test)), np.ones(len(one_test))])

# Choosing alpha = 3 for high accuracy
alpha = 3

# Estimating parameters using training data
beta_hat = 1 / (alpha * len(zero_train)) * np.sum(zero_train) 
mu_hat = 1 / len(one_train) * np.sum(one_train) 
sigma_squared_hat = np.var(one_train)

print(f'\u03B2 = {beta_hat:.3}', f'\u03BC = {mu_hat:.3}', 
      f'\u03C3\u00B2 = {sigma_squared_hat:.3}', sep='\n')

# Defining distributions
def gamma_func(alpha):
    return np.math.factorial(alpha - 1)

def gamma_dist(x):
    return 1 / (beta_hat**alpha * gamma_func(alpha)) * x**(alpha - 1) * np.exp(-x / beta_hat)

def normal_dist(x):
    return 1 / (np.sqrt(2 * np.pi * sigma_squared_hat)) * np.exp(
        -((x - mu_hat) ** 2) / (2 * sigma_squared_hat))

# Generating the x-values
x_g = np.linspace(np.min(zero_values), np.max(zero_values), int(1e4))
x_n = np.linspace(np.min(one_values), np.max(one_values), int(1e4))

# Computing the Gamma and Normal distributions
y_gamma = gamma_dist(x_g)  # Gamma distribution for class 0 (zero_values)
y_norm = normal_dist(x_n)  # Normal distribution for class 1 (one_values)

n_bins = 40 

# Plotting histograms and distributions
if __name__ == '__main__':
    plt.hist(zero_values, bins=n_bins, density=True, alpha=0.5, label='zeroes', color='green', edgecolor='grey')
    plt.hist(one_values, bins=n_bins, density=True, alpha=0.5, label='ones', color='yellow', edgecolor='grey')
    plt.plot(x_g, y_gamma, label='Gamma Distribution', color='darkgreen')
    plt.plot(x_n, y_norm, label='Gauss Distribution', color='goldenrod')
    plt.xlabel('Values')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.title('Histograms and distributions')
    plt.show()

# Prior-probabilities
P_zero = len(zero_train) / (len(zero_train) + len(one_train))
P_one = len(one_train) / (len(zero_train) + len(one_train))

# Bayesian classifier logic based on given row data
def bayes_classifier(feature_value):
    if P_zero * gamma_dist(feature_value) > P_one * normal_dist(feature_value):
        return 0  # Classified as zero
    else:
        return 1  # Classified as one

# Classifying the test set and get predicted labels
predicted_labels = np.array([bayes_classifier(val) for val in test_values])

# Creating confusion matrix counters
true_zero = 0
false_zero = 0
true_one = 0
false_one = 0

# Iterate over test data to calculate confusion matrix
for true_label, predicted_label in zip(test_labels, predicted_labels):
    if true_label == 0:
        if predicted_label == 0:
            true_zero += 1 
        else:
            false_zero += 1
    else:
        if predicted_label == 1:
            true_one += 1 
        else:
            false_one += 1 

# Constructing the confusion matrix
cm = np.array([[true_zero, false_zero], [false_one, true_one]])

# Plotting confusion matrix
if __name__ == '__main__':
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Predicted zeroes', 'Predicted ones'],
                yticklabels=['Actual zeroes', 'Actual ones'])
    plt.title('Confusion Matrix for Bayes Classifier')
    plt.show()

N = true_zero + false_zero + false_one + true_one

accuracy = (true_zero + true_one)/N
precision = true_zero/(true_zero + false_zero)
recall = true_zero/(true_zero + false_one)

print(f'Test accuracy:{accuracy * 100:.2f}%', f'precision: {precision * 100:.2f}%', 
      f'recall: {recall * 100:.2f}%', sep='\n')