# Python (pytorch) 

###### more coding on jupyter note 

## Items:
Main Dataset **Ref.** (for neural decoding):
* Makin, Joseph G., et al. "Superior arm-movement decoding from cortex with a new, unsupervised-learning algorithm." Journal of neural engineering 15.2 (2018)

# Environment
* Python version = 3.6
* Ubuntu = 16.04
* pytorch_gpu = True

# Main Issue
*To decode nonhuman neural signal, datasets is  from Sabes lab.*
```
Paper:
Makin, Joseph G., et al. "Superior arm-movement decoding from cortex with a new, unsupervised-learning algorithm." Journal of neural engineering 15.2 (2018)

Decoder used this datasets code:
single session decode model maybe could ref. 'read_data_20_07_16_v1.py'
cross session model(CNN + GRU) : 2020_08_17_cross_session_baseline.ipynb


Analysis coefficient of correlation with 'neural signal' &&  'velocity':
cc_analysis_modify.ipynb


Observation:
Attention module on GAN's generator
compare nn.embedding / nn.Linear / positional emb. with input 'label(condition)'
```
## 解決問題
```
Result:
對比作者使用卡爾曼濾波的修改model(recurrent exponential-family harmonium,rEFH)，
透過實驗使用卡爾曼濾波對於cross data是無法幫助提升動作預測的準確
主因大多在於神經變異問題(如下述附表圖1)
透過使用深度學習的方式，解決此一問題，使得可使用更大量的train data來提升準確
預測R_square比較，黑色為作者的(圖2
並且
由於time bin區間會影響其預測，透過實驗找出有效的time bin區間，達到最佳預測(圖3

嘗試使用GAN的方式去擴增神經資料(暫無提升幫助)(圖4

```

實驗，根據假設(off line)的情況:

透過cross data的方式，使得訓練資料變多，進而提升預測值

不同天數的神經細胞數據，會有差異性問題，ex:速度(vel)所對應的spike firing的量根據不同天的數據有可能會有不同

如圖1所示 第7筆實驗數據與第11筆的spike firing明顯不同(圖1)
![image](https://github.com/jp298486/nc_lab/blob/master/base_model/image/spike_firing_variation.png)

如圖2所示，使用深度學習的方式對於single data與cross data的模型對比作者的預測值
![image](https://github.com/jp298486/nc_lab/blob/master/base_model/image/result_with_cross_data_predict.png)

如圖3所示，找出最佳的time bin區間

![image](https://github.com/jp298486/nc_lab/blob/master/base_model/image/spike_with_movement.png)
![image](https://github.com/jp298486/nc_lab/blob/master/base_model/image/anilysis_data_move_with_time.png)

如圖4所示，透過GAN的方式合成神經訊號
![image](https://github.com/jp298486/nc_lab/blob/master/base_model/image/synthesis_data_by_gan_test.png)
