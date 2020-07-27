import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import random, math, os, time

import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd

from models.GlobalAttention import Global_Attention, Encoder, Decoder, Seq2Seq

from utils.early_stopping import EarlyStopping
from utils.prepare_Traffic import test_traffic_single_direction
from utils.support import *
from utils.adamw import AdamW
from utils.cyclic_scheduler import CyclicLRWithRestarts

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# set the random seeds for reproducability
SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, optimizer, criterion, X_train, y_train):
    # model.train()

    iter_per_epoch = int(np.ceil(X_train.shape[0] * 1. / BATCH_SIZE))
    iter_losses = np.zeros(EPOCHS * iter_per_epoch)

    n_iter = 0

    perm_idx = np.random.permutation(X_train.shape[0])

    # train for each batch

    for t_i in range(0, X_train.shape[0], BATCH_SIZE):
        batch_idx = perm_idx[t_i:(t_i + BATCH_SIZE)]

        x_train_batch = np.take(X_train, batch_idx, axis=0)
        y_train_batch = np.take(y_train, batch_idx, axis=0)

        loss = train_iteration(model, optimizer, criterion, CLIP,
                               x_train_batch, y_train_batch)

        if t_i % 50 == 0:
            print('batch_loss:{}'.format(loss))

        iter_losses[t_i // BATCH_SIZE] = loss

        n_iter += 1

    return np.mean(iter_losses[range(0, iter_per_epoch)])


def train_iteration(model, optimizer, criterion, clip, X_train, y_train):
    model.train()
    optimizer.zero_grad()

    X_train = np.transpose(X_train, [1, 0, 2])
    y_train = np.transpose(y_train, [1, 0, 2])

    X_train_tensor = numpy_to_tvar(X_train)
    y_train_tensor = numpy_to_tvar(y_train)

    output = model(X_train_tensor, y_train_tensor)

    output = output.view(-1)

    y_train_tensor = y_train_tensor.view(-1)

    loss = criterion(output, y_train_tensor)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

    optimizer.step()

    scheduler.batch_step()

    return loss.item()


### evaluate


def evaluate(model, criterion, X_test, y_test, scaler_x, scaler_y):
    # model.eval()

    epoch_loss = 0
    iter_per_epoch = int(np.ceil(X_test.shape[0] * 1. / BATCH_SIZE))
    iter_losses = np.zeros(EPOCHS * iter_per_epoch)
    # other loss: MAE RMSLE
    iter_multiloss = [
        np.zeros(EPOCHS * iter_per_epoch),
        np.zeros(EPOCHS * iter_per_epoch),
        np.zeros(EPOCHS * iter_per_epoch)
    ]
    iter_losses = np.zeros(EPOCHS * iter_per_epoch)
    perm_idx = np.random.permutation(X_test.shape[0])

    n_iter = 0

    with torch.no_grad():
        for t_i in range(0, X_test.shape[0], BATCH_SIZE):
            batch_idx = perm_idx[t_i:(t_i + BATCH_SIZE)]

            x_test_batch = np.take(X_test, batch_idx, axis=0)
            y_test_batch = np.take(y_test, batch_idx, axis=0)

            loss, mae, rmsle, rmse = evaluate_iteration(
                model, criterion, x_test_batch, y_test_batch, scaler_x,
                scaler_y)
            iter_losses[t_i // BATCH_SIZE] = loss
            iter_multiloss[0][t_i // BATCH_SIZE] = mae
            iter_multiloss[1][t_i // BATCH_SIZE] = rmsle
            iter_multiloss[2][t_i // BATCH_SIZE] = rmse

            n_iter += 1

    return np.mean(iter_losses[range(0, iter_per_epoch)]), np.mean(
        iter_multiloss[0][range(0, iter_per_epoch)]), np.mean(
            iter_multiloss[1][range(0, iter_per_epoch)]), np.mean(
                iter_multiloss[2][range(0, iter_per_epoch)])


def evaluate_iteration(model, criterion, x_test, y_test, scaler_x, scaler_y):
    model.eval()

    x_test = np.transpose(x_test, [1, 0, 2])
    y_test = np.transpose(y_test, [1, 0, 2])

    x_test_tensor = numpy_to_tvar(x_test)
    y_test_tensor = numpy_to_tvar(y_test)

    output = model(x_test_tensor, y_test_tensor, 0)

    output = output.view(-1)
    y_test_tensor = y_test_tensor.view(-1)

    loss = criterion(output, y_test_tensor)

    # metric
    output_numpy = output.cpu().data.numpy()
    y_test_numpy = y_test_tensor.cpu().data.numpy()

    output_numpy = scaler_y.inverse_transform(output_numpy.reshape(-1, 1))
    y_test_numpy = scaler_y.inverse_transform(y_test_numpy.reshape(-1, 1))

    loss_mae = mean_absolute_error(y_test_numpy, output_numpy)
    loss_RMSLE = np.sqrt(mean_squared_error(y_test_numpy, output_numpy))
    loss_RMSE = np.sqrt(mean_squared_error(y_test_numpy, output_numpy))

    return loss.item(), loss_mae, loss_RMSLE, loss_RMSE


if __name__ == "__main__":

    # model hyperparameters
    INPUT_DIM = 52
    OUTPUT_DIM = 1
    ENC_HID_DIM = 25
    DEC_HID_DIM = 25
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1
    ECN_Layers = 2
    DEC_Layers = 2
    LR = 0.001  # learning rate
    CLIP = 1
    EPOCHS = 500
    BATCH_SIZE = 20

    ## Different test data

    (X_train, y_train, x_train_len,
     x_train_before_len), (X_test, y_test, x_test_len, x_test_before_len), (
         scaler_x, scaler_y) = test_traffic_single_direction()

    print('\nsize of x_train, y_train, x_test, y_test:')
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # Model
    glob_attn = Global_Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(INPUT_DIM, ENC_HID_DIM, DEC_HID_DIM, ECN_Layers, DEC_Layers,
                  ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_Layers,
                  DEC_DROPOUT, glob_attn)

    model = Seq2Seq(enc, dec, device).to(device)
    model.apply(init_weights)

    print(model)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    # Adam
    optimizer = AdamW(model.parameters(), lr=1e-2, weight_decay=1e-5)
    scheduler = CyclicLRWithRestarts(optimizer,
                                     BATCH_SIZE,
                                     17062,
                                     restart_period=5,
                                     t_mult=1.2,
                                     policy="cosine")

    criterion = nn.MSELoss()

    # Early Stopping
    # initialize the early_stopping object
    # early stopping patience; how long to wait after last time validation loss improved.
    patience = 10
    early_stopping = EarlyStopping(output_path='Decoder/checkpoints/Traffic_N0_3.pt',
                                   patience=patience,
                                   verbose=True)



    # Training

    best_valid_loss = float('inf')
    for epoch in range(EPOCHS):

        train_epoch_losses = np.zeros(EPOCHS)
        evaluate_epoch_losses = np.zeros(EPOCHS)

        print('Epoch:', epoch)

        scheduler.step()

        start_time = time.time()
        train_loss = train(model, optimizer, criterion, X_train, y_train)
        valid_loss, _, _, _ = evaluate(model, criterion, X_test, y_test,
                                       scaler_x, scaler_y)
        end_time = time.time()

        train_epoch_losses[epoch] = train_loss
        evaluate_epoch_losses[epoch] = valid_loss

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(
            f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}'
        )
        print(
            f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}'
        )

    # prediction

    
    # model.load_state_dict(torch.load('checkpoints/checkpoint.pt',map_location='cpu'))
    
    # test_loss, test_mae, test_rmsle, test_rmse = evaluate(model, criterion, X_test, y_test, scaler_x, scaler_y)
    
    # # plt.show()
    
    # print(f'| Test Loss: {test_loss:.4f} | Test PPL: {math.exp(test_loss):7.4f} |')
    # print(f'| MAE: {test_mae:.4f} | Test PPL: {math.exp(test_mae):7.4f} |')
    # print(f'| RMSLE: {test_rmsle:.4f} | Test PPL: {math.exp(test_rmsle):7.4f} |')
    # print(f'| RMSE: {test_rmse:.4f} | Test PPL: {math.exp(test_rmse):7.4f} |')