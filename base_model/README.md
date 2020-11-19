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

>> single session decode model maybe could ref. 'read_data_20_07_16_v1.py'

>> cross session model(CNN + GRU) : 2020_08_17_cross_session_baseline.ipynb

Analysis coefficient of correlation with 'neural signal' &&  'velocity':

>> cc_analysis_modify.ipynb

Observation:
Attention module on GAN's generator
compare nn.embedding / nn.Linear / positional emb. with input 'label(condition)'
```
