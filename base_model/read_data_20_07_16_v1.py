import os
from os import listdir
import h5py
import math
import time
import argparse

import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import r2_score
'''
ArgumentParser()
metavar='' 默認提示會顯示大寫 所以用這個方式不顯示或另取名稱
有使用action 就不用另外標示

此為single session的訓練

'''
dataset_path = '/home/nclab62159/machineLearning/plot/train_data_source/'

if os.path.isdir(dataset_path) != True:
    print('\n>> Not find dataset folder path: '+dataset_path)
    print('>> Please Check folder_path && Computer working system !\n')

# argument setting
parser = argparse.ArgumentParser(description='Read nerual data(single session); if change args , use --cmd \' \'.')

parser.add_argument('--dataroot',type=str,
    default=dataset_path, help = 'default :'+dataset_path, metavar='')
parser.add_argument('--batch_size_train', type=int, default=32, help='input batch_size for training (default:32)', metavar='')
parser.add_argument('--batch_size_r2_score', type=int, default=128, help='input batch_size for testing (default:128)', metavar='')
parser.add_argument('--epoch', type=int, default =20, help='default:20')
parser.add_argument('--file_name', action='store_true', help='show file name in folder.')
parser.add_argument('--ch_count', type=int, default=0, help='select channel count.0=ch96 ,1=ch100.(default :0)', metavar='')
parser.add_argument('--bin_width', type=int, default=16, help='ori. data is 4ms,using 64ms. (default :16)', metavar='')
parser.add_argument('--select_session', type=int, default=0, help='which session be observed. (default :0)', metavar='')
parser.add_argument('--select_label', type=str, default='vel', help='enter string for vel/pos/acc. (default:vel)', metavar='')
parser.add_argument('--normal', type=str, default='std', help='normalization for movement, choice std/max.(default:std)', metavar='')
parser.add_argument('--plot_session_info', action='store_true', help='plot session Info for Fr/finger_pos/finger_vel/finger_acc')
parser.add_argument('--order', type=int, default=4, help='firing rate use Order count (default:4)', metavar='')
parser.add_argument('--train_count', type=int, default=5000, help='train count (default:0)')
parser.add_argument('--train', action='store_true', help='start training')
parser.add_argument('--predict', action='store_true', help='load model and get pred r2_score')
args = parser.parse_args() # 參數宣告後使用
# print(args.batch_size_train)
DATE = time.strftime('%Y_%m_%d',time.localtime())

# use cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


Channel_List = []

def Get_File_Name():
    List_File = listdir(args.dataroot)
    List_File.sort()
    print('\n',List_File[:])
    
    
if args.ch_count ==0 or args.ch_count ==1 :
    if args.ch_count == 0:
        print('use channel 96')
        for kk in range(96):
            Channel_List.append(kk+1)
        # Channel_List = np.array(Channel_List)

    if args.ch_count == 1:
        print('use channel 100')
        channel_arr = []
        channel_arr.append(['N', 42, 46, 25, 31, 35, 39, 41, 47, 'N'])
        channel_arr.append([38, 40, 48, 27, 29, 33, 37, 43,  6, 45])
        channel_arr.append([34, 36, 44,  1,  9, 13, 17, 21,  2, 88])
        channel_arr.append([30, 32, 89, 93,  5, 15, 19, 23,  8, 84])
        channel_arr.append([26, 28, 81, 85, 87, 91,  7,  4, 86, 80])
        channel_arr.append([22, 24, 77, 79, 83,  3, 11, 66, 82, 76])
        channel_arr.append([18, 20, 73, 75, 95, 54, 62, 74, 78, 72])
        channel_arr.append([14, 16, 94, 96, 57, 58, 50, 70, 64, 68])
        channel_arr.append([10, 12, 90, 92, 61, 65, 69, 71, 56, 60])
        channel_arr.append([ 'N', 51, 49, 53, 55, 59, 63, 67, 52 ,'N' ])
        for i in range(10):
            for j in range(10):
                Channel_List.append(channel_arr[i][j])
        # Channel_List = np.array(Channel_List)
            
else:
    print('ch_counterror, please select 0 or 1')

if args.file_name :
    Get_File_Name()
def Get_Spike_Firing(firing_point, bins):
    '''
    將取得的firing 時間轉換成計數
    input array is spike firing time;  used this to count amount
    ex: 
        以 indy_20160407_02.mat 為例
        t最後一筆為882.78,但是,
        計算 firing 的點為 887.7254（hash_unit 的最後一筆）
        所以只計算時間內的firing 次數
    '''
    spike_data = []
    time_last_point = bins[-1]
    
    for kk in firing_point:
        if kk < time_last_point:
            spike_data.append(kk)
    mapping_data = np.digitize(spike_data, bins)
    map_data, bin_arr = np.histogram(spike_data, bins = bins)
    
    return map_data.reshape(-1, len(map_data))

# init. var
channel = []
finger_pos_z = []
finger_pos_x = []
finger_pos_y = []

vel_z = []
vel_x = []
vel_y = []

acc_z = []
acc_x = []
acc_y = []

Firing_rate_hash = []
Firing_rate_unit_1 = []
Firing_rate_unit_2 = []
Firing_rate_unit_3 = []
Firing_rate_unit_4 = []

Unsort_data = []
Sorting_data = []

Session_train_pos = []
Session_train_vel = []
Session_train_acc = []

Session_test_pos = []
Session_test_vel = []
Session_test_acc = []

Log_train_loss = []
Log_test_loss = []

# ----------------------------------------------------------
List_File = listdir(args.dataroot)
List_File.sort()
file_name = List_File[args.select_session]
print('Dataset :',args.dataroot+str(file_name))
mat_file = h5py.File(args.dataroot+str(file_name), 'r') # read mat file
CHANNELS = mat_file[list(mat_file.keys())[1]] # electric channel Info
CURSOR_POS = mat_file[list(mat_file.keys())[2]] # cursor position
FINGER_POS = mat_file[list(mat_file.keys())[3]] # finger position
SPIKES = mat_file[list(mat_file.keys())[4]] # Spike firing time point 這邊是紀錄發生spike的時間點,所以根據bins寬度,要另外整理
Session_Unit = SPIKES.shape[0] # session unit count ,此一session有幾個被sorting的unit 第一筆皆為hash
TIMES = mat_file[list(mat_file.keys())[5]] # session t 0.004=4ms

time_bin = (TIMES[0])[::args.bin_width]
print('\n == mat file Info == :')
print('Mat File keys()', list(mat_file.keys()))
print('Channel count :', CHANNELS)
print('Cursor position :', CURSOR_POS)
print('Finger position :', FINGER_POS)
print('Spike  :', SPIKES)
print('Time :', TIMES)
print(' == Session Info. == :')
print('Session Unit count :', Session_Unit)
print('observered time bin ('+str(0.004*args.bin_width*1000)+' ms):')
print('>>',time_bin)
print('>> time count:',time_bin.shape, ' ; Trial time ', time_bin[-1]- time_bin[0])

channel = Channel_List

# finger position shape = time shape
finger_pos_z = FINGER_POS[0][::args.bin_width]
finger_pos_x = FINGER_POS[1][::args.bin_width]
finger_pos_y = FINGER_POS[2][::args.bin_width]
print('>> Finger Position shape :', finger_pos_z.shape)
# get velocity cm => mm
vel_z = (finger_pos_z[1:] - finger_pos_z[:-1])*(-10)/(0.004*args.bin_width)
vel_x = (finger_pos_x[1:] - finger_pos_x[:-1])*(-10)/(0.004*args.bin_width)
vel_y = (finger_pos_y[1:] - finger_pos_y[:-1])*(-10)/(0.004*args.bin_width)
print('>> Velocity shape :', vel_z.shape)
# get accelerate
acc_z = (vel_z[1:] - vel_z[:-1])/ (0.004*args.bin_width)
acc_x = (vel_x[1:] - vel_x[:-1])/ (0.004*args.bin_width)
acc_y = (vel_y[1:] - vel_y[:-1])/ (0.004*args.bin_width)
print('>> Accelerate shape :', acc_z.shape)
acc_z = np.concatenate((np.zeros(1), acc_z), axis=0) # be the same shape for vel ,so add zero shape 
acc_x = np.concatenate((np.zeros(1), acc_x), axis=0) 
acc_y = np.concatenate((np.zeros(1), acc_y), axis=0) 

# get firing rate
search_index = 0
for ch_index in channel:
    '''
    If use channel count 100 , the channel list has 'N'
    ch_index-1 : because setting channel number start is 1 ,but program is start 0
    '''
    # concat data by channel
    fr_unit_hash = []
    fr_unit_01 = []
    fr_unit_02 = []
    fr_unit_03 = []
    fr_unit_04 = []
    
    if str(ch_index) != 'N':
        
        unit_hash = mat_file[SPIKES[0][ch_index-1]] 
        unit_01 = mat_file[SPIKES[1][ch_index-1]]
        unit_02 = mat_file[SPIKES[2][ch_index-1]]

        # unit.shape[0] == 2 ,.mat file show this is null array>>表示 .mat檔對應的是空集合的channel
        if unit_hash.shape[0] != 2 :
            fr_unit_hash = Get_Spike_Firing(unit_hash[0], time_bin)
        else:
            fr_unit_hash = np.zeros([1,time_bin.shape[0]-1])

        if unit_01.shape[0] != 2 :
            fr_unit_01 = Get_Spike_Firing(unit_01[0], time_bin)
        else:
            fr_unit_01 = np.zeros([1,time_bin.shape[0]-1]) 

        if unit_02.shape[0] !=2 :    
            fr_unit_02 = Get_Spike_Firing(unit_02[0], time_bin)
        else:
            fr_unit_02 = np.zeros([1,time_bin.shape[0]-1])

        if Session_Unit == 5:
            unit_03 = mat_file[SPIKES[3][ch_index-1]]
            unit_04 = mat_file[SPIKES[4][ch_index-1]]
            if unit_03.shape[0] !=2 : 
                fr_unit_03 = Get_Spike_Firing(unit_03[0], time_bin)
            else:
                fr_unit_03 = np.zeros([1,time_bin.shape[0]-1])

            if unit_04.shape[0] !=2 :
                fr_unit_04 = Get_Spike_Firing(unit_04[0], time_bin)
            else:
                fr_unit_04 = np.zeros([1,time_bin.shape[0]-1])
        else:
            fr_unit_03 = np.zeros([1,time_bin.shape[0]-1])
            fr_unit_04 = np.zeros([1,time_bin.shape[0]-1])
        
    else:
        fr_unit_hash = np.zeros([1,time_bin.shape[0]-1])
        fr_unit_01 = np.zeros([1,time_bin.shape[0]-1])
        fr_unit_02 = np.zeros([1,time_bin.shape[0]-1])
        fr_unit_03 = np.zeros([1,time_bin.shape[0]-1])
        fr_unit_04 = np.zeros([1,time_bin.shape[0]-1])
    # concate unit data

    if search_index == 0:
        Firing_rate_hash = fr_unit_hash
        Firing_rate_unit_1 = fr_unit_01
        Firing_rate_unit_2 = fr_unit_02
        Firing_rate_unit_3 = fr_unit_03
        Firing_rate_unit_4 = fr_unit_04
    else:
        Firing_rate_hash = np.concatenate((Firing_rate_hash, fr_unit_hash), axis=0)
        Firing_rate_unit_1 = np.concatenate((Firing_rate_unit_1, fr_unit_01), axis=0)
        Firing_rate_unit_2 = np.concatenate((Firing_rate_unit_2, fr_unit_02), axis=0)
        Firing_rate_unit_3 = np.concatenate((Firing_rate_unit_3, fr_unit_03), axis=0)
        Firing_rate_unit_4 = np.concatenate((Firing_rate_unit_4, fr_unit_04), axis=0)
    search_index +=1
'''
valid data: use 
ex: np.all(Firing_rate_unit_4 == 0)
if this session not have 5 unit
Firing_rate_unit_3 && Firing_rate_unit_4 is a zeros array
'''
# print('=====',np.all(Firing_rate_unit_4 == 0))
Unsort_data = Firing_rate_hash+ Firing_rate_unit_1+ Firing_rate_unit_2+ Firing_rate_unit_3 +Firing_rate_unit_4
Sorting_data = np.concatenate((Firing_rate_hash.reshape(1, Firing_rate_hash.shape[0], Firing_rate_hash.shape[1]),
                               Firing_rate_unit_1.reshape(1, Firing_rate_unit_1.shape[0], Firing_rate_unit_1.shape[1]),
                               Firing_rate_unit_2.reshape(1, Firing_rate_unit_2.shape[0], Firing_rate_unit_2.shape[1]),
                               Firing_rate_unit_3.reshape(1, Firing_rate_unit_3.shape[0], Firing_rate_unit_3.shape[1]),
                               Firing_rate_unit_4.reshape(1, Firing_rate_unit_4.shape[0], Firing_rate_unit_4.shape[1])),axis = 0)
print('Unsort Data shape:', Unsort_data.shape)
print('Sorting Data shape :', Sorting_data.shape)
# model

class Fc_Net(nn.Module):
    def __init__(self, ch, order):
        super(Fc_Net, self).__init__()
        self.ch_count = ch
        self.order_count = order+1
        
        self.fc = nn.Linear(self.ch_count*self.order_count,64)
        self.pReLu = nn.PReLU()
        self.fc_x_1 = nn.Linear(64, 32)
        self.fc_x_2 = nn.Linear(32, 1)
        

        self.fc_y_1 = nn.Linear(64, 32)
        self.fc_y_2 = nn.Linear(32, 1)
        
    def forward(self, Data):
        data = Data.view(-1, self.ch_count*self.order_count)
        
        feature = self.pReLu(self.fc(data))

        x = torch.tanh(self.fc_x_1(feature))
        out_x = (self.fc_x_2(x))

        y = torch.tanh(self.fc_y_1(feature))
        out_y = (self.fc_y_2(y))
        return out_x.squeeze(), out_y.squeeze()


def train(model, train_loader, optimizer, epoch):
    # .train()配合.eval() ,一次epoch 後有使用eval()檢查測試loss,那麼下個epoch跑之前,必須使用.train()
    # jupyter 有不同寫法
    model.train() 
    for n, (Data, Label) in enumerate(train_loader):
        optimizer.zero_grad()
        fr_data = Data
        valid_x = Label[:, -1, 0]
        valid_y = Label[:, -1, 1]
        loss_mse = nn.MSELoss()
        if device:
            fr_data = fr_data.to(device)
            valid_x = valid_x.to(device)
            valid_y = valid_y.to(device)
        out_x, out_y = model(fr_data)
        loss_x = loss_mse(out_x, valid_x)
        loss_y = loss_mse(out_y, valid_y)
        
        loss = loss_x+ loss_y
        if n % 30 == 0:
            Log_train_loss.append(loss)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        print('Train Epoch[{}], loss:{:.4f} >> loss_x:{:.4f}, loss_y:{:.4f}'.format(
            epoch+1, loss.item(), loss_x.item(), loss_y.item()))
def test(model, test_loader,epoch):
    model.eval()
    with torch.no_grad():
        for n, (Data, Label) in enumerate(test_loader):
            fr_data = Data
            valid_x = Label[:, -1, 0]
            valid_y = Label[:, -1, 1]
            loss_mse = nn.MSELoss()
            if device:
                fr_data = fr_data.to(device)
                valid_x = valid_x.to(device)
                valid_y = valid_y.to(device)
            out_x, out_y = model(fr_data)
            loss_x = loss_mse(out_x, valid_x)
            loss_y = loss_mse(out_y, valid_y)
            loss = loss_x+ loss_y
            if n % 3 == 0:
                Log_test_loss.append(loss)
        print('== Test Data loss == loss:{:.4f} >> loss_x:{:.4f}, loss_y:{:.4f}'.format( loss.item()
        , loss_x.item(), loss_y.item()))
def plot_loss_log(train_data_loss, test_data_loss):
    train_loss = np.array(train_data_loss)
    test_loss = np.array(test_data_loss)
    p_x = []
    for kk in range(len(train_data_loss)):
        p_x.append(kk)
    p_x = np.array(p_x)
    plt.figure(figsize=(10,10))
    ax_loss = plt.gca()
    ax_loss.set_title('Loss for train/test data', size=25)
    ax_loss.plot(p_x, train_loss, label='train_loss')
    ax_loss.plot(p_x, test_loss, label='test loss')
    ax_loss.legend(fontsize=20)
    plt.savefig(os.path.splitext(file_name)[0]+'_'+str(DATE)+'_'+args.select_label+'_Loss.png')

def plot_predict(real_data, pred_data, r2_score):
    xp = []
    for kk in range(real_data.shape[0]):
        xp.append(kk)
    xp = np.array(xp)
    bound_down = 0
    bound_up = len(xp)
    fig,ax = plt.subplots(real_data.shape[1], 1, figsize=(10,12))
    ax[0].set_title(args.select_label+' R$^{2}$_x:' + str(r2_score[0]), size = 20)
    ax[0].plot(xp, real_data[:,0], label='real')
    ax[0].plot(xp, pred_data[:,0], label='pred')
    ax[0].set_xlim(bound_down, bound_up)
    ax[0].legend(fontsize=20)

    ax[1].set_title(args.select_label+' R$^{2}$_y:' + str(r2_score[1]), size = 20)
    ax[1].plot(xp, real_data[:,1], label='real')
    ax[1].plot(xp, pred_data[:,1], label='pred')
    ax[1].set_xlim(bound_down, bound_up)
    ax[1].legend(fontsize=20)
    plt.savefig(os.path.splitext(file_name)[0]+'_'+str(DATE)+'_'+args.select_label+'_predict.png')
# --------------------Prepare Training Data/Label && Testing Data/Label --------------------
fr = np.transpose(Unsort_data)

px = finger_pos_x[1:]
px = px.reshape(len(px), 1) # reshape for concat data to label
py = finger_pos_y[1:]
py = py.reshape(len(py), 1)
pz = finger_pos_z[1:]
pz = pz.reshape(len(pz), 1)

vx = vel_x
vx = vx.reshape(len(vx), 1)
vy = vel_y
vy = vy.reshape(len(vy), 1)
vz = vel_z
vz = vz.reshape(len(vz), 1)

ax = acc_x
ax = ax.reshape(len(ax), 1)
ay = acc_y
ay = ay.reshape(len(ay), 1)
az = acc_z
az = az.reshape(len(az), 1)

train_fr = fr[:args.train_count]
var_fr = fr[args.train_count:args.train_count+500]
test_fr = fr[args.train_count:]

train_px = px[:args.train_count]
train_py = py[:args.train_count]
train_pz = pz[:args.train_count]

train_vx = vx[:args.train_count]
train_vy = vy[:args.train_count]
train_vz = vz[:args.train_count]

train_ax = ax[:args.train_count]
train_ay = ay[:args.train_count]
train_az = az[:args.train_count]

test_px = px[args.train_count:]
test_py = py[args.train_count:]
test_pz = pz[args.train_count:]

test_vx = vx[args.train_count:]
test_vy = vy[args.train_count:]
test_vz = vz[args.train_count:]

test_ax = ax[args.train_count:]
test_ay = ay[args.train_count:]
test_az = az[args.train_count:]
# every session testing data not be same, so using var_data to check model training
var_px = px[args.train_count:args.train_count+500]
var_py = py[args.train_count:args.train_count+500]
var_pz = pz[args.train_count:args.train_count+500]

var_vx = vx[args.train_count:args.train_count+500]
var_vy = vy[args.train_count:args.train_count+500]
var_vz = vz[args.train_count:args.train_count+500]

var_ax = ax[args.train_count:args.train_count+500]
var_ay = ay[args.train_count:args.train_count+500]
var_az = az[args.train_count:args.train_count+500]

# normalize data :
# fr >> use min-max & fr_min = 0,so fr/(fr_max-fr_min) ,because fr not have negative
# movement >> std , (data-mean)/std
train_fr_normal = train_fr/(np.max(train_fr) - np.min(train_fr))
test_fr_normal = var_fr/(np.max(train_fr) - np.min(train_fr))
real_fr_normal = test_fr/(np.max(train_fr) - np.min(train_fr))
if args.normal == 'std':
    # train
    train_px_normal = (train_px - np.mean(train_px))/np.std(train_px)
    train_py_normal = (train_py - np.mean(train_py))/np.std(train_py)
    train_pz_normal = (train_pz - np.mean(train_pz))/np.std(train_pz)

    train_vx_normal = (train_vx - np.mean(train_vx))/np.std(train_vx)
    train_vy_normal = (train_vy - np.mean(train_vy))/np.std(train_vy)
    train_vz_normal = (train_vz - np.mean(train_vz))/np.std(train_vz)

    train_ax_normal = (train_ax - np.mean(train_ax))/np.std(train_ax)
    train_ay_normal = (train_ay - np.mean(train_ay))/np.std(train_ay)
    train_az_normal = (train_az - np.mean(train_az))/np.std(train_az)

    Session_train_pos = np.hstack((train_px_normal, train_py_normal, train_pz_normal))
    Session_train_vel = np.hstack((train_vx_normal, train_vy_normal, train_vz_normal))
    Session_train_acc = np.hstack((train_ax_normal, train_ay_normal, train_az_normal))
    # test 
    test_px_normal = (var_px - np.mean(train_px))/np.std(train_px)
    test_py_normal = (var_py - np.mean(train_py))/np.std(train_py)
    test_pz_normal = (var_pz - np.mean(train_pz))/np.std(train_pz)

    test_vx_normal = (var_vx - np.mean(train_px))/np.std(train_px)
    test_vy_normal = (var_vy - np.mean(train_py))/np.std(train_py)
    test_vz_normal = (var_vz - np.mean(train_pz))/np.std(train_pz)

    test_ax_normal = (var_ax - np.mean(train_px))/np.std(train_px)
    test_ay_normal = (var_ay - np.mean(train_py))/np.std(train_py)
    test_az_normal = (var_az - np.mean(train_pz))/np.std(train_pz)

    Session_test_pos = np.hstack((test_px_normal, test_py_normal, test_pz_normal))
    Session_test_vel = np.hstack((test_vx_normal, test_vy_normal, test_vz_normal))
    Session_test_acc = np.hstack((test_ax_normal, test_ay_normal, test_az_normal))


elif args.normal == 'max':
    # train
    train_px_normal = (train_px - np.min(train_px))/(np.max(train_px) - np.min(train_px))
    train_py_normal = (train_py - np.min(train_py))/(np.max(train_py) - np.min(train_py))
    train_pz_normal = (train_pz - np.min(train_pz))/(np.max(train_pz) - np.min(train_pz))

    train_vx_normal = (train_vx - np.min(train_vx))/(np.max(train_vx) - np.min(train_vx))
    train_vy_normal = (train_vy - np.min(train_vy))/(np.max(train_vy) - np.min(train_vy))
    train_vz_normal = (train_vz - np.min(train_vz))/(np.max(train_vz) - np.min(train_vz))

    train_ax_normal = (train_ax - np.min(train_ax))/(np.max(train_ax) - np.min(train_vx))
    train_ay_normal = (train_ay - np.min(train_ay))/(np.max(train_ay) - np.min(train_vy))
    train_az_normal = (train_az - np.min(train_az))/(np.max(train_az) - np.min(train_vz))

    Session_train_pos = np.hstack((train_px_normal, train_py_normal, train_pz_normal))
    Session_train_vel = np.hstack((train_vx_normal, train_vy_normal, train_vz_normal))
    Session_train_acc = np.hstack((train_ax_normal, train_ay_normal, train_az_normal))
    # test
    test_px_normal = (var_px - np.min(train_px))/(np.max(train_px) - np.min(train_px))
    test_py_normal = (var_py - np.min(train_py))/(np.max(train_py) - np.min(train_py))
    test_pz_normal = (var_pz - np.min(train_pz))/(np.max(train_pz) - np.min(train_pz))

    test_vx_normal = (var_vx - np.min(train_vx))/(np.max(train_vx) - np.min(train_vx))
    test_vy_normal = (var_vy - np.min(train_vy))/(np.max(train_vy) - np.min(train_vy))
    test_vz_normal = (var_vz - np.min(train_vz))/(np.max(train_vz) - np.min(train_vz))

    test_ax_normal = (var_ax - np.min(train_ax))/(np.max(train_ax) - np.min(train_ax))
    test_ay_normal = (var_ay - np.min(train_ay))/(np.max(train_ay) - np.min(train_ay))
    test_az_normal = (var_az - np.min(train_az))/(np.max(train_az) - np.min(train_az))

    Session_test_pos = np.hstack((test_px_normal, test_py_normal, test_pz_normal))
    Session_test_vel = np.hstack((test_vx_normal, test_vy_normal, test_vz_normal))
    Session_test_acc = np.hstack((test_ax_normal, test_ay_normal, test_az_normal))
else:
    print(' >> ERROR : for select normalize method .\n')

train_label = []
var_label = []
if args.select_label == 'pos':
    train_label = Session_train_pos
    var_label = Session_test_pos
elif args.select_label == 'vel':
    train_label = Session_train_vel
    var_label = Session_test_vel
elif args.select_label == 'acc': 
    train_label = Session_train_acc
    var_label = Session_test_acc
else:
    print(' >> ERROR : for select label')
# ------- add order
train_fr_order = []
test_fr_order = []
real_fr_order = []

train_label_order = []
test_label_order = []

if args.order == 0:
    train_fr_order = train_fr_normal.reshape(train_fr_normal.shape[0], 1, train_fr_normal.shape[1])
    test_fr_order = test_fr_normal.reshape(test_fr_normal.shape[0], 1, test_fr_normal.shape[1])
    real_fr_order = real_fr_normal.reshape(real_fr_normal.shape[0], 1, real_fr_normal.shape[1])

    train_label_order = train_label.reshape(train_label.shape[0], 1, train_label.shape[1])
    test_label_order = var_label.reshape(var_label.shape[0], 1, var_label.shape[1])
else:
    # train_fr_normal
    pre_order = [] 
    pre_data = []
    pre_order = np.zeros([args.order , train_fr_normal.shape[1]])
    pre_data = np.concatenate((pre_order, train_fr_normal), axis = 0)
    for kk in range(args.order):
        n = pre_data[kk :(kk-args.order)]
        data = n.reshape(n.shape[0], 1, n.shape[1])
        if kk == 0:
            train_fr_order = data
        else:
            train_fr_order = np.concatenate((train_fr_order, data), axis=1)
    train_fr_order = np.concatenate((train_fr_order, 
                                    train_fr_normal.reshape(train_fr_normal.shape[0], 1, train_fr_normal.shape[1])), axis=1)
    # test_fr_normal
    pre_order = [] 
    pre_data = []
    pre_order = np.zeros([args.order , test_fr_normal.shape[1]])
    pre_data = np.concatenate((pre_order, test_fr_normal), axis = 0)
    for kk in range(args.order):
        n = pre_data[kk :(kk-args.order)]
        data = n.reshape(n.shape[0], 1, n.shape[1])
        if kk == 0:
            test_fr_order = data
        else:
            test_fr_order = np.concatenate((test_fr_order, data), axis=1)
    test_fr_order = np.concatenate((test_fr_order,
                                    test_fr_normal.reshape(test_fr_normal.shape[0], 1, test_fr_normal.shape[1])), axis=1)
    
    # real_fr_normal
    pre_order = [] 
    pre_data = []
    pre_order = np.zeros([args.order , real_fr_normal.shape[1]])
    pre_data = np.concatenate((pre_order, real_fr_normal), axis = 0)
    for kk in range(args.order):
        n = pre_data[kk :(kk-args.order)]
        data = n.reshape(n.shape[0], 1, n.shape[1])
        if kk == 0:
            real_fr_order = data
        else:
            real_fr_order = np.concatenate((real_fr_order, data), axis=1)
    real_fr_order = np.concatenate((real_fr_order,
                                    real_fr_normal.reshape(real_fr_normal.shape[0], 1, real_fr_normal.shape[1])), axis=1)

    # train_label
    pre_order = [] 
    pre_data = []
    pre_order = np.zeros([args.order , train_label.shape[1]])
    pre_data = np.concatenate((pre_order, train_label), axis = 0)
    for kk in range(args.order):
        n = pre_data[kk :(kk-args.order)]
        data = n.reshape(n.shape[0], 1, n.shape[1])
        if kk == 0:
            train_label_order = data
        else:
            train_label_order = np.concatenate((train_label_order, data), axis=1)
    train_label_order = np.concatenate((train_label_order,
                                    train_label.reshape(train_label.shape[0], 1, train_label.shape[1])), axis=1)
    # var_label
    pre_order = [] 
    pre_data = []
    pre_order = np.zeros([args.order , var_label.shape[1]])
    pre_data = np.concatenate((pre_order, var_label), axis = 0)
    for kk in range(args.order):
        n = pre_data[kk :(kk-args.order)]
        data = n.reshape(n.shape[0], 1, n.shape[1])
        if kk == 0:
            test_label_order = data
        else:
            test_label_order = np.concatenate((test_label_order, data), axis=1)
    test_label_order = np.concatenate((test_label_order,
                                    var_label.reshape(var_label.shape[0], 1, var_label.shape[1])), axis=1)
#-------------------------------------------------------------------
# dataset && dataloader
train_data = torch.from_numpy(train_fr_order).type(torch.FloatTensor)
train_label = torch.from_numpy(train_label_order).type(torch.FloatTensor)
train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size = args.batch_size_train, shuffle=True)

test_data = torch.from_numpy(test_fr_order).type(torch.FloatTensor)
test_label = torch.from_numpy(test_label_order).type(torch.FloatTensor)
test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size = args.batch_size_train, shuffle =True)

model = Fc_Net(ch=len(Channel_List),order=args.order)
optim = torch.optim.Adam( model.parameters(), lr = 0.0001)
if device:
    model.to(device)
# Train? path = os.getcwd()
if args.train:
     
    if os.path.isdir('models') != True:
        # check save folder exist , if not be created, 
        # 假如不給指定路徑 都會生成在當前code 的資料夾內
        os.mkdir('models')
    # 類似c的寫法 ％s = string , ％d = 整數输出, %f = 浮點數
    save_model_path = '%s/Fc_model_%s_epoch_%d_%s.pth' %('models',DATE,args.epoch,args.select_label)
    for epoch in range(args.epoch):
        train(model, train_dataloader, optim, epoch)
        test(model, test_dataloader,epoch)
    
    plot_loss_log(Log_train_loss, Log_test_loss)

    torch.save(model.state_dict(),save_model_path)

#plot session infomation
if args.plot_session_info :
    
    fig, ax_session = plt.subplots(4, 1, figsize=(30, 20))
    plot_x = []
    for k in range(acc_z.shape[0]):
        plot_x.append(k)
    plot_x = np.array(plot_x)
    # can change bound range to obversered
    bound_left = 0
    bound_Right = plot_x.shape[0]
    
    ax_session[0].set_title(str(file_name), size=30)
    ax_session[0].plot(plot_x, finger_pos_x[1:], color='r', label='x')
    ax_session[0].plot(plot_x, finger_pos_y[1:], color='b', label='y')
    ax_session[0].plot(plot_x, finger_pos_z[1:], color='g', label='z')
    ax_session[0].set_ylabel('Position(mm)', size=25)
    ax_session[0].set_xlim(bound_left, bound_Right)
    ax_session[0].legend(fontsize=20, loc=1)

    ax_session[1].plot(plot_x, vel_x, color='r', label='x')
    ax_session[1].plot(plot_x, vel_y, color='b', label='y')
    ax_session[1].plot(plot_x, vel_z, color='g', label='z')
    ax_session[1].set_ylabel('Velocity(mm/s)', size=25)
    ax_session[1].set_xlim(bound_left, bound_Right)
    ax_session[1].legend(fontsize=20, loc=1)

    ax_session[2].plot(plot_x, acc_x, color='r', label='x')
    ax_session[2].plot(plot_x, acc_y, color='b', label='y')
    ax_session[2].plot(plot_x, acc_z, color='g', label='z')
    ax_session[2].set_ylabel('Accelerate(mm/s$^{2}$)', size=25)
    ax_session[2].set_xlim(bound_left, bound_Right)
    ax_session[2].legend(fontsize=20, loc=1)

    sns.heatmap(Unsort_data,cmap='viridis',cbar =False, cbar_kws= {'orientation' : 'vertical'},
    xticklabels=False, yticklabels=False, vmin=0, vmax=4,ax=ax_session[3])
    ax_session[3].set_ylabel('FiringRate', size=25)
    ax_session[3].set_xlim(bound_left, bound_Right)
    plt.savefig(os.path.splitext(file_name)[0]+'_'+str(DATE)+'.png')

if args.predict :
    with torch.no_grad():
        # load model has to new a model && load model info.
        test_model = Fc_Net(ch=len(Channel_List),order=args.order) 
        test_model.load_state_dict(torch.load('%s/Fc_model_%s_epoch_%d_%s.pth' %('models',DATE,args.epoch,args.select_label)))
        if device:
            test_model.to(device)
        real_data = torch.from_numpy(real_fr_order).type(torch.FloatTensor)
        # not use order for real label , because it is not used for calculate loss
        # 這邊隨便給 label 因為不會使用到回傳loss 值
        real_label = torch.from_numpy(test_px).type(torch.FloatTensor) 
        real_dataset = torch.utils.data.TensorDataset(real_data, real_label)
        # not shuffle 不使用隨機,因為要對比實際運動
        real_dataloader = torch.utils.data.DataLoader(dataset = real_dataset, batch_size = args.batch_size_r2_score, shuffle=False)
        
        pred_x = []
        pred_y = []
        real_x = []
        real_y = []
        for n, (Data, Label) in enumerate(real_dataloader):
            real_fr = Data
            if device:
                real_fr = real_fr.to(device)
                
            out_x, out_y = test_model(real_fr)
            if n == 0 :
                pred_x = out_x.data.cpu().numpy()
                pred_y = out_y.data.cpu().numpy()
            else :
                pred_x = np.concatenate((pred_x, out_x.data.cpu().numpy()), axis = 0)
                pred_y = np.concatenate((pred_y, out_y.data.cpu().numpy()), axis = 0)
        if args.select_label == 'pos': 
            real_x = test_px[:, 0]
            real_y = test_py[:, 0]
        elif args.select_label == 'vel':
            real_x = test_vx[:, 0]
            real_y = test_vy[:, 0]
        elif args.select_label == 'acc':
            real_x = test_ax[:, 0]
            real_y = test_ay[:, 0]

        if args.normal == 'std':
            if args.select_label == 'pos':
                pred_x = pred_x*np.std(train_px) +  np.mean(train_px)
                pred_y = pred_y*np.std(train_py) +  np.mean(train_py)
            elif args.select_label == 'vel':
                pred_x = pred_x*np.std(train_vx) +  np.mean(train_vx)
                pred_y = pred_y*np.std(train_vy) +  np.mean(train_vy)
            elif args.select_label == 'acc':
                pred_x = pred_x*np.std(train_ax) +  np.mean(train_ax)
                pred_y = pred_y*np.std(train_ay) +  np.mean(train_ay)
            else:
                pass
        elif args.normal == 'max':
            if args.select_label == 'pos':
                pred_x = pred_x*(np.max(train_px) - np.min(train_px)) +  np.min(train_px)
                pred_y = pred_y*(np.max(train_py) - np.min(train_py)) +  np.min(train_py)
            elif args.select_label == 'vel':
                pred_x = pred_x*(np.max(train_vx) - np.min(train_vx)) +  np.min(train_vx)
                pred_y = pred_y*(np.max(train_vy) - np.min(train_vy)) +  np.min(train_vy)
            elif args.select_label == 'acc':
                pred_x = pred_x*(np.max(train_ax) - np.min(train_ax)) +  np.min(train_ax)
                pred_y = pred_y*(np.max(train_az) - np.min(train_az)) +  np.min(train_ay)
            else:
                pass
        else:
            pass
        
        print('>> '+args.select_label+' r2_x',np.round(r2_score(real_x, pred_x),4))
        print('>> '+args.select_label+' r2_y',np.round(r2_score(real_y, pred_y),4))
        r2x = np.round(r2_score(real_x, pred_x),4)
        r2y = np.round(r2_score(real_y, pred_y),4)
        
        real_data = np.concatenate( ( real_x.reshape(real_x.shape[0], 1), real_y.reshape(real_y.shape[0], 1)), axis=1 )
        pred_data = np.concatenate( ( pred_x.reshape(pred_x.shape[0], 1), pred_y.reshape(pred_y.shape[0], 1)), axis=1 )
        # print(np.hstack((r2x, r2y)).shape)
        # print(real_data.shape)
        # print(pred_data.shape)
        plot_predict( real_data, pred_data,np.hstack((r2x, r2y)) )
