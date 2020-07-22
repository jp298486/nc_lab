快速上手
使用python 3.6 linux環境 open .mat file

step1:
變更程式內的dataset的路徑,顯示的路徑為linux的,需要變更放置dataset的位置

step2:
使用cmd/terminal cd到檔案路徑
輸入
python read_data_20_07_16_v1.py -h

step3:
可以指定dataset底下的檔案,已在程式內寫上排序,所以會跟日期排序有關
假如只有一筆dataset .mat也可執行
python read_data_20_07_16_v1.py --train
