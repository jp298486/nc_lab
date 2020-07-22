快速上手
使用python 3.6 linux環境 open .mat file

step1:
變更程式內的dataset的路徑,顯示的路徑為linux的,需要變更放置dataset的位置

step2:
使用cmd/terminal cd到檔案路徑
輸入
python read_data_20_07_16_v1.py -h
>> 顯示提示如下
Read nerual data(single session); if change args , use --cmd ' '.

optional arguments:
  -h, --help            show this help message and exit
  --dataroot            default :/home/nclab62159/machineLearning/plot/train_d
                        ata_source/
  --batch_size_train    input batch_size for training (default:32)
  --batch_size_r2_score 
                        input batch_size for testing (default:128)
  --epoch EPOCH         default:20
  --file_name           show file name in folder.
  --ch_count            select channel count.0=ch96 ,1=ch100.(default :0)
  --bin_width           ori. data is 4ms,using 64ms. (default :16)
  --select_session      which session be observed. (default :0)
  --select_label        enter string for vel/pos/acc. (default:vel)
  --normal              normalization for movement, choice
                        std/max.(default:std)
  --plot_session_info   plot session Info for
                        Fr/finger_pos/finger_vel/finger_acc
  --order               firing rate use Order count (default:4)
  --train_count TRAIN_COUNT
                        train count (default:0)
  --train               start training
  --predict             load model and get pred r2_score

step3:
可以指定dataset底下的檔案,已在程式內寫上排序,所以會跟日期排序有關
假如只有一筆dataset .mat也可執行
python read_data_20_07_16_v1.py --train
