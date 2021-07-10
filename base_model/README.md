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
根據假設，off line的情況:

透過cross data的方式，使得訓練資料變多，進而提升預測值

不同天數的神經細胞數據，會有差異性問題，ex:速度(vel)所對應的spike firing的量根據不同天的數據有可能會有不同
## 解決問題
根據假設，off line的情況:

透過cross data的方式，使得訓練資料變多，進而提升預測值

不同天數的神經細胞數據，會有差異性問題，ex:速度(vel)所對應的spike firing的量根據不同天的數據有可能會有不同
![image.](https://github.com/jp298486/nc_lab/blob/master/base_model/image/spike_firing_variation.png)
