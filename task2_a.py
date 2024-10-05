import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# a)

# Imorting data
file_path = 'data_problem2.csv' 
data = pd.read_csv(file_path)

data_transposed = data.T
values = data_transposed.index.astype(float)

# Identifying which zeroes and ones correspond to different values
zero_values = values[data_transposed[0] == 0]
one_values = values[data_transposed[0] == 1]

n_bins = 40

# Plotting histograms
if __name__ == '__main__':
    plt.hist(zero_values, bins=n_bins, alpha=0.5, label='zeroes', color='green', edgecolor='grey')
    plt.hist(one_values, bins=n_bins, alpha=0.5, label='ones', color='yellow', edgecolor='grey')
    plt.title('Values in classes 0 and 1')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()