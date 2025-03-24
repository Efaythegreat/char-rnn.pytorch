#!/usr/bin/env python
# https://github.com/spro/char-rnn.pytorch

import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os

from tqdm import tqdm

import matplotlib.pyplot as plt


from helpers import *
from model import *
from generate import *

# import nltk.corpus.words

# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('filename', type=str)
argparser.add_argument('--model', type=str, default="gru")
argparser.add_argument('--n_epochs', type=int, default=2000)
argparser.add_argument('--print_every', type=int, default=100)
argparser.add_argument('--hidden_size', type=int, default=100)
argparser.add_argument('--n_layers', type=int, default=2)
argparser.add_argument('--learning_rate', type=float, default=0.01)
argparser.add_argument('--chunk_len', type=int, default=200)
argparser.add_argument('--batch_size', type=int, default=100)
argparser.add_argument('--shuffle', action='store_true')
argparser.add_argument('--cuda', action='store_true')

# hyperparameter_name
argparser.add_argument('--name', type=str, default=None)
argparser.add_argument('--value', type=float, default=None)
argparser.add_argument('--save_file', type=str, default=None)
args = argparser.parse_args()

if args.cuda:
    print("Using CUDA")

if args.save_file is None:
    args.save_file = args.filename

# print(args.chunk_len)
# print(f"Chunk length: {len(chunk)}, Expected: {args.chunk_len}")

# creates csv for graphs
############################################################################################
import time

log_csv_path = 'training_log.csv'
hyper_parameter_error_path = 'hyper_parameter_error.csv'

def log_hyperparameters_over_time_to_csv(hiddensize, n_layers, chunk_len, batch_size, learning_rate, total_time):
    # Create a DataFrame with the hyperparameters
    df = pd.DataFrame({
        'hidden_size': [hiddensize],
        'n_layers': [n_layers],
        'chunk_len': [chunk_len],
        'batch_size': [batch_size],
        'learning_rate': [learning_rate],
        'total_time': [total_time]
    })

    # Append the DataFrame to the CSV file
    if os.path.exists(log_csv_path):
        df.to_csv(log_csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(log_csv_path, mode='w', header=True, index=False)


def error_over_epoch_csv(hyperparameter, error_list, folder='error_logs', parameter_value=None):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    csv_path = os.path.join(folder, f'{hyperparameter}_error.csv')
    column_name = f"{hyperparameter}_{parameter_value}"
    df_new = pd.DataFrame({column_name: error_list[:2000]})
    

    # Append the DataFrame to the CSV file
    
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        df_combined = pd.concat([df_old, df_new], axis=1)
        
        # os.remove(csv_path)
        
    else:
        df_combined = df_new
        
    df_combined.to_csv(csv_path, mode='w', header=True, index=False)
    
########################################################################################################

file, file_len = read_file(args.filename)

def random_training_set(chunk_len, batch_size):
    inp = torch.LongTensor(batch_size, chunk_len) # inp is input tensor
    target = torch.LongTensor(batch_size, chunk_len) # target is target tensor
    for bi in range(batch_size):
        start_index = random.randint(0, file_len - chunk_len - 1) # bi is batch index
        # end_index = start_index + chunk_len + 1
        end_index = start_index + chunk_len + 1
        chunk = file[start_index:end_index]
        
        inp[bi] = char_tensor(chunk[:-1])
        target[bi] = char_tensor(chunk[1:])
        
        
        # inp[bi] = char_tensor(chunk[:chunk_len])
        # target[bi] = char_tensor(chunk[1:chunk_len + 1])
        
    inp = Variable(inp)
    target = Variable(target)
    if args.cuda:
        inp = inp.cuda()
        target = target.cuda() 
    return inp, target

def train(inp, target):
    hidden = decoder.init_hidden(args.batch_size)
    if args.cuda:
        hidden = hidden.cuda()
    decoder.zero_grad()
    loss = 0

    for c in range(args.chunk_len):
        output, hidden = decoder(inp[:,c], hidden)
        loss += criterion(output.view(args.batch_size, -1), target[:,c])

    loss.backward()
    decoder_optimizer.step()

    # return loss.data[0] / args.chunk_len
    return loss.item() / args.chunk_len

# def save():
#     save_filename = os.path.splitext(os.path.basename(args.filename))[0] + '.pt'
#     torch.save(decoder, save_filename)
#     print('Saved as %s' % save_filename)
    
folder_for_saving = 'saved_models'    

def save():
    # args.safe_file = 'default_model' without .pt
    save_filename = os.path.splitext(os.path.basename(args.save_file))[0] + '.pt'
    path = os.path.join(folder_for_saving, save_filename)
    
    if not os.path.exists(folder_for_saving): # check if the folder exists
        os.makedirs(folder_for_saving)
        
    torch.save(decoder, path)
    print('Saved as %s' % path)


# Initialize models and start training

decoder = CharRNN( 
    n_characters,
    args.hidden_size,
    n_characters,
    model=args.model,
    n_layers=args.n_layers,
)

# above is the model, below is the optimizer

decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()

if args.cuda:
    decoder.cuda()

import pandas as pd

start = time.time()
all_losses = []
loss_avg = 0

try:
    all_losses = []
    
    print("Training for %d epochs..." % args.n_epochs)
    for epoch in tqdm(range(1, args.n_epochs + 1)):
        loss = train(*random_training_set(args.chunk_len, args.batch_size))
        loss_avg += loss
        
        all_losses.append(loss)

        # if epoch % args.print_every + 400 == 0:
            
        if epoch % 500 == 0:
            print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / args.n_epochs * 100, loss))
            print(generate(decoder, 'Wh', 100, cuda=args.cuda), '\n')

    total_time_min = (time.time() - start) / 60
    log_hyperparameters_over_time_to_csv(args.hidden_size, args.n_layers, args.chunk_len, args.batch_size, args.learning_rate, total_time_min)
    
    # Save the hyperparameters and their corresponding errors to a CSV file
    error_over_epoch_csv(hyperparameter=args.name, error_list=all_losses, parameter_value=args.value)
    
    # Save the model
    
    
    print("Saving...")
    save()
    
    
    
    
    # path = 'default_loss_bigger_chunk'
    # path_png = path + '.png'
    # path_csv = path + '.csv'
    
    # if path exists, remove it
    # if os.path.exists(path_csv):
    #     os.remove(path_csv)
        
    # if os.path.exists(path_png):
    #     os.remove(path_png)
    
    # df = pd.DataFrame({'epoch': range(len(all_losses)), 'loss': all_losses})
    # df.to_csv(path_csv, index=False)
    
    # plt.plot(all_losses, label='Training loss')
    # plt.xlabel('Iterations')
    # plt.ylabel('Loss')
    # plt.title('Training loss over time')
    # plt.legend()
    # plt.savefig(path_png)
    # plt.show()

except KeyboardInterrupt:
    print("Saving before quit...")
    save()

# the input encoding happens at line 60
# the output encoding happens at line 61
# the loss happens at line 62
# the optimizer happens at line 63
# the training happens at line 64
# the model happens at line 65