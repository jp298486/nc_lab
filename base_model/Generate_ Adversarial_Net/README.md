python 3.6 ;linux system

step 1:
一開始先看 11/02與10/31
透過自定義的分布使用GAN的方式去擬合這些分布

step 2:
9/15的主要使用MNIST的dataset
主要比較
原始gan

cGAN
cGAN的label使用何種方式轉換成vector
[1] nn.embedding
[2] nn.Linear
[3] postional embedding
Attention:
主要對於G的label部份做self-attention
以及
noise concat label的attention
ps:
網路的self-attention GAN 是基於sn-gan的框架
