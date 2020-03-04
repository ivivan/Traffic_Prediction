import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import random, math, os, time

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from utils.adamw import AdamW
from utils.cyclic_scheduler import CyclicLRWithRestarts

from utils.early_stopping import EarlyStopping
from utils.prepare_WL import test_pm25_single_station

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# set the random seeds for reproducability
SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########## Support


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def numpy_to_tvar(x):
    return Variable(torch.from_numpy(x).type(torch.FloatTensor).to(device))


def series_to_superviesed(x_timeseries,
                          y_timeseries,
                          n_memory_step,
                          n_forcast_step,
                          split=None):
    '''
        x_timeseries: input time series data, numpy array, (time_step, features)
        y_timeseries: target time series data,  numpy array, (time_step, features)
        n_memory_step: number of memory step in supervised learning, int
        n_forcast_step: number of forcase step in supervised learning, int
        split: portion of data to be used as train set, float, e.g. 0.8
    '''
    assert len(x_timeseries.shape
               ) == 2, 'x_timeseries must be shape of (time_step, features)'
    assert len(y_timeseries.shape
               ) == 2, 'y_timeseries must be shape of (time_step, features)'

    input_step, input_feature = x_timeseries.shape
    output_step, output_feature = y_timeseries.shape
    assert input_step == output_step, 'number of time_step of x_timeseries and y_timeseries are not consistent!'

    n_RNN_sample = input_step - n_forcast_step - n_memory_step + 1
    RNN_x = np.zeros((n_RNN_sample, n_memory_step, input_feature))
    RNN_y = np.zeros((n_RNN_sample, n_forcast_step, output_feature))

    for n in range(n_RNN_sample):
        RNN_x[n, :, :] = x_timeseries[n:n + n_memory_step, :]
        RNN_y[n, :, :] = y_timeseries[n + n_memory_step:n + n_memory_step +
                                      n_forcast_step, :]
    if split != None:
        assert (split <= 0.9) & (split >= 0.1), 'split not in reasonable range'
        return RNN_x[:int(split * len(RNN_x))], RNN_y[:int(split * len(RNN_x))], \
               RNN_x[int(split * len(RNN_x)) + 1:], RNN_y[int(split * len(RNN_x)) + 1:]
    else:
        return RNN_x, RNN_y, None, None


########### Model


class Encoder(nn.Module):
    def __init__(self, input_dim, enc_hid_dim, dec_hid_dim, enc_layers,
                 dec_layers, dropout_p):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.dropout_p = dropout_p

        self.input_linear = nn.Linear(self.input_dim, self.enc_hid_dim)
        self.lstm = nn.LSTM(input_size=self.enc_hid_dim,
                            hidden_size=self.enc_hid_dim,
                            num_layers=self.enc_layers,
                            bidirectional=True)
        self.output_linear = nn.Linear(self.enc_hid_dim * 2, self.dec_hid_dim)
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, input):

        embedded = self.dropout(torch.tanh(self.input_linear(input)))

        outputs, (hidden, cell) = self.lstm(embedded)

        hidden = torch.tanh(
            self.output_linear(
                torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        # for different number of decoder layers
        hidden = hidden.repeat(self.dec_layers, 1, 1)

        return outputs, (hidden, hidden)


class Global_Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super(Global_Attention, self).__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn = nn.Linear(self.enc_hid_dim * 2 + self.dec_hid_dim,
                              self.dec_hid_dim)
        self.v = nn.Parameter(torch.rand(self.dec_hid_dim))

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # only pick up last layer hidden
        hidden = torch.unbind(hidden, dim=0)[0]

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        energy = torch.tanh(
            self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        energy = energy.permute(0, 2, 1)

        v = self.v.repeat(batch_size, 1).unsqueeze(1)

        attention = torch.bmm(v, energy).squeeze(1)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, enc_hid_dim, dec_hid_dim, dec_layers,
                 dropout_p, attention):
        super(Decoder, self).__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dec_layers = dec_layers
        self.dropout_p = dropout_p
        self.attention = attention

        self.input_dec = nn.Linear(self.output_dim, self.dec_hid_dim)
        self.lstm = nn.LSTM(input_size=self.enc_hid_dim * 2 + self.dec_hid_dim,
                            hidden_size=self.dec_hid_dim,
                            num_layers=self.dec_layers)
        self.out = nn.Linear(
            self.enc_hid_dim * 2 + self.dec_hid_dim + self.dec_hid_dim,
            self.output_dim)
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, input, hidden, cell, encoder_outputs):

        input = input.unsqueeze(0)
        input = torch.unsqueeze(input, 2)

        embedded = self.dropout(torch.tanh(self.input_dec(input)))

        a = self.attention(hidden, encoder_outputs)

        a = a.unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        lstm_input = torch.cat((embedded, weighted), dim=2)

        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        input_dec = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        output = self.out(torch.cat((output, weighted, input_dec), dim=1))

        return output.squeeze(1), (hidden, cell), a


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):

        batch_size = src.shape[1]
        max_len = trg.shape[0]

        outputs = torch.zeros(max_len, batch_size,
                              self.decoder.output_dim).to(self.device)

        decoder_attn = torch.zeros(max_len, src.shape[0]).to(self.device)

        encoder_outputs, (hidden, cell) = self.encoder(src)

        # only use y initial y
        output = src[-1, :, 0]

        for t in range(0, max_len):

            output, (hidden,
                     cell), attn_weight = self.decoder(output, hidden, cell,
                                                       encoder_outputs)

            outputs[t] = output.unsqueeze(1)

            teacher_force = random.random() < teacher_forcing_ratio

            output = (trg[t].view(-1) if teacher_force else output)

        return outputs


def train(model, optimizer, criterion, X_train, y_train):

    iter_per_epoch = int(np.ceil(X_train.shape[0] * 1. / BATCH_SIZE))
    iter_losses = np.zeros(EPOCHS * iter_per_epoch)

    n_iter = 0

    perm_idx = np.random.permutation(X_train.shape[0])

    # train for each batch

    for t_i in range(0, X_train.shape[0], BATCH_SIZE):
        batch_idx = perm_idx[t_i:(t_i + BATCH_SIZE)]

        x_train_batch = np.take(X_train, batch_idx, axis=0)
        y_train_batch = np.take(y_train, batch_idx, axis=0)

        loss = train_iteration(model, optimizer, criterion, CLIP, WD,
                               x_train_batch, y_train_batch)

        if t_i % 50 == 0:
            print('batch_loss:{}'.format(loss))

        iter_losses[t_i // BATCH_SIZE] = loss

        n_iter += 1

    return np.mean(iter_losses[range(0, iter_per_epoch)])


def train_iteration(model, optimizer, criterion, clip, wd, X_train, y_train):
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

            # writer.add_scalars('Val_loss', {'val_loss': iter_losses[t_i // BATCH_SIZE]},
            #                    n_iter)

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
    loss_RMSLE = RMSLE(y_test_numpy, output_numpy)
    loss_RMSE = np.sqrt(mean_squared_error(y_test_numpy, output_numpy))

    return loss.item(), loss_mae, loss_RMSLE, loss_RMSE


if __name__ == "__main__":

    INPUT_DIM = 1
    OUTPUT_DIM = 1
    ENC_HID_DIM = 25
    DEC_HID_DIM = 25
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1
    ECN_Layers = 2
    DEC_Layers = 2
    LR = 0.001  # learning rate
    WD = 0.1  # weight decay
    CLIP = 1
    EPOCHS = 1000
    BATCH_SIZE = 100

    (x_train, y_train, x_train_len,
     x_train_before_len), (x_test, y_test, x_test_len, x_test_before_len), (
         scaler_x, scaler_y) = test_pm25_single_station()

    print('\nsize of x_train, y_train, x_test, y_test:')
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    # time series to image

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

    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = CyclicLRWithRestarts(optimizer,
                                     BATCH_SIZE,
                                     68673,
                                     restart_period=5,
                                     t_mult=1.2,
                                     policy="cosine")

    criterion = nn.MSELoss()

    # Early Stopping
    # initialize the early_stopping object
    # early stopping patience; how long to wait after last time validation loss improved.
    patience = 10
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    best_valid_loss = float('inf')
    for epoch in range(EPOCHS):

        train_epoch_losses = np.zeros(EPOCHS)
        evaluate_epoch_losses = np.zeros(EPOCHS)

        print('Epoch:', epoch)

        scheduler.step()

        start_time = time.time()
        train_loss = train(model, optimizer, criterion, x_train, y_train)
        valid_loss, _, _, _ = evaluate(model, criterion, x_test, y_test,
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

    # # prediction
    #
    # #
    # model.load_state_dict(torch.load('checkpoint.pt',map_location='cpu'))
    #
    # test_loss, test_mae, test_rmsle, test_rmse = evaluate(model, criterion, x_test, y_test, scaler_x, scaler_y)
    #
    # # plt.show()
    #
    # print(f'| Test Loss: {test_loss:.4f} | Test PPL: {math.exp(test_loss):7.4f} |')
    # print(f'| MAE: {test_mae:.4f} | Test PPL: {math.exp(test_mae):7.4f} |')
    # print(f'| RMSLE: {test_rmsle:.4f} | Test PPL: {math.exp(test_rmsle):7.4f} |')
    # print(f'| RMSE: {test_rmse:.4f} | Test PPL: {math.exp(test_rmse):7.4f} |')
