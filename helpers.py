# https://github.com/spro/char-rnn.pytorch

import unidecode
import string
import random
import time
import math
import torch

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Reading and un-unicode-encoding data

all_characters = string.printable
n_characters = len(all_characters)

def read_file(filename):
    file = unidecode.unidecode(open(filename).read())
    return file, len(file)

# Turning a string into a tensor

def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        try:
            tensor[c] = all_characters.index(string[c])
        except:
            continue
    return tensor

# Readable time elapsed

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


###############################################################
# new functions from yifei

# def plot_training_time(df, title, x, y, x_label, num_ticks=10):
#     df.plot(kind='bar', x=x, y=y, title=title, legend=True)
#     plt.xlabel(x_label)
#     plt.ylabel('Training time (minutes)')
#     plt.xticks(rotation=45, ha='right')
#     plt.show()
    
#     # save the plot as png
#     plt.savefig(title + '.png')
    
#     plt.show()
    
    
# def moving_average(data, window_size=10):
#     return data.rolling(window=window_size, min_periods=1).mean()

    
# def plot_training_error(df, title):
    
#     smoothed_df = df.copy()
#     for col in df.columns[1:]:
#         smoothed_df[col] = moving_average(df[col], window_size=10)
    
#     smoothed_df.plot(kind='line', title=title, legend=True)
#     plt.xlabel('Epoch')
#     plt.ylabel('Training error')
    
#     # save the plot as png
#     plt.savefig(title + '.png')
#     plt.show()