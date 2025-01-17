# Data_mining_Project2
Implementation of sentiment analysis with IMDB data
contributed by <`williamchang`>
參考書籍：『Tensorflow+Keras 深度學習人工智慧實務應用』
#### 資料夾內容
- IMDB_decision_tree.ipnb: Decision tree implementation (For homework, including data creation and decition tree)
- IMDB_review.ipnb: Using ML method to recongnize data (For homework and Final representation)
- README.md: This file represent
- Untitled.ipnb: Test code for another target(Not for homework)
### 資料介紹
- 資料來源：[IMDb review datasets from ACL 2011 paper](http://ai.stanford.edu/~amaas/data/sentiment/)
- 動機：資源容易取得，且對於資料的前處理部分找到了不錯的方法進行篩選，因為資料特性，是文字的預測，對於比賽能有不錯的學習分類方式，因此，選取這個資料集當作我的資料集
- 資料介紹：資料即是採用 IMDb 的電影網路資料集，是雅馬遜公司旗下的網站，在美國是最多人瀏覽的電影相關網站，已累積相當多的電影資料，經過整理過後，會分成訓練資料及預測資料評論資料（包含正面及負面資料），各為 25,000 個，分成訓練資料及測試資料（合計 50,000 筆資料）。
- 方法概念：經由這些處理過的資料，呈現每個單字的出現次數，經由這些單字每個都給一個 token，建立成一個 hash table，並給每一個字的權重，使用 (Word embedding) 自然語言處理的方法，嘗試建立各種模型（如:MLP、RNN 和 LSTM），去預測準確度，這幾個方法在準確度方面相差不多，在建立模型當中，會介紹為何選取該模型進行建立，並把模型建立過程詳細介紹。也會藉由建立好的模型，對現行的資料進行預測（採用剛上映的電影：Ralph Breaks the Internet），分類預測正面及負面共兩種。
- 資料分別在 test 集 train 資料內
	- ('read', 'train', 'files:', 25000)
	- ('read', 'test', 'files:', 25000)
### 實作介紹
- train_text:0 ~ 12499 都是正面評論的文字
	- 12500 ~ 24999 都是負面評論的文字
- y_train:0 ~ 12499 正面評價都為1
	- 12500 ~ 24999 負面評價都是0
- 如下圖呈現正面及負面資料範例呈現
![image](https://github.com/williamchangTW/Data_mining_Project2/blob/master/%E8%9E%A2%E5%B9%95%E5%BF%AB%E7%85%A7%202018-12-03%20%E4%B8%8B%E5%8D%885.02.21.png)

- 關鍵：把資料轉換成數字的 HASH TABLE 進行儲存，型態為 Dictionary，用於查找 token 用
~~~
[444, 19, 31, 775, 61, 304, 283, 1, 1647, 4, 31, 295, 1037, 622, 655, 2, 488, 67, 2, 55, 12, 15, 37, 1926, 3, 5, 15, 37, 795, 464, 36, 3, 48, 322, 5, 75, 241, 35, 8, 28, 17, 50, 37, 12, 7, 55, 96, 20, 559, 1, 700, 15, 37, 795, 2, 12, 914, 5, 786, 141, 362, 55, 96, 75, 9, 1093, 35, 1, 295, 5, 7, 3, 247, 155, 55, 96, 42, 55, 558, 3, 1183, 79, 1, 15, 3, 9, 800, 1553, 138, 7, 144, 1649, 17, 160, 7, 68, 9, 12, 1649, 542, 1081, 28, 604, 1888, 2, 55, 12, 1055, 7, 3, 991, 39, 50, 8, 605, 36, 55, 65, 2, 3, 1370, 55, 215, 37, 75, 2, 320, 7, 3, 329, 55, 868, 5, 1779, 35, 1, 723, 1370, 2, 349, 5, 86, 15, 53, 193, 4, 760, 37, 109, 12, 7, 129, 6, 3, 251, 33, 37, 109, 5, 603, 156, 6, 303, 2, 20, 39, 5, 164, 29, 55, 44, 3, 303, 478, 10, 6, 31, 61, 71, 22, 11, 6, 1, 27, 11, 109, 206, 937]
~~~
- 把資料置換成同一長度（長度 ＝ 100），不夠填 0，超過直接捨去開頭幾個
	- 以下這個例子為超過 100
~~~
('before pad_sequences length=', 111)
[1158, 185, 16, 1057, 15, 799, 1585, 17, 30, 299, 4, 1313, 13, 3, 180, 17, 639, 15, 3, 1821, 33, 6, 5, 986, 14, 37, 30, 1, 5, 604, 1, 135, 15, 22, 51, 69, 1991, 1, 1305, 224, 6, 399, 6, 1216, 13, 17, 50, 1096, 79, 3, 943, 30, 3, 19, 1, 346, 1865, 179, 62, 376, 1, 582, 3, 2, 374, 22, 3, 172, 2, 6, 83, 249, 13, 3, 564, 1249, 1, 16, 6, 750, 3, 1654, 4, 892, 2, 1, 17, 47, 3, 444, 19, 1, 114, 30, 1, 6, 364, 4, 835, 121, 69, 30, 163, 484, 33, 3, 273, 15, 296, 237, 35]
~~~
- 刪除過後
~~~
('after pad_sequences length=', 100)
[1313   13    3  180   17  639   15    3 1821   33    6    5  986   14
   37   30    1    5  604    1  135   15   22   51   69 1991    1 1305
  224    6  399    6 1216   13   17   50 1096   79    3  943   30    3
   19    1  346 1865  179   62  376    1  582    3    2  374   22    3
  172    2    6   83  249   13    3  564 1249    1   16    6  750    3
 1654    4  892    2    1   17   47    3  444   19    1  114   30    1
    6  364    4  835  121   69   30  163  484   33    3  273   15  296
  237   35]
~~~
- 只有正面資料的單字出現排名（前 50 名）
~~~
1.the	2.and	3.a	4.of	5.to	6.is	7.in	8.it	9.i	10.this	
11.that	12.as	13.with	14.for	15.was	16.but	17.film	18.movie19.his	20.on
21.are	22.he	23.you	24.not	25.one	26.have	27.be	28.by	29.all	30.an
31.at	32.who	33.from	34.her	35.they	36.has	37.so	38.like	39.it's	40.about
41.very	42.out	43.or	44.good	45.more	46.when	47.some	48.what	49.if	50.just
~~~
- 正面資料圖示
![image](https://github.com/williamchangTW/Data_mining_Project2/blob/master/%E8%9E%A2%E5%B9%95%E5%BF%AB%E7%85%A7%202018-12-03%20%E4%B8%8B%E5%8D%886.59.16.png)
- 只有負面資料的單字出現排名（前 50 名）
~~~
1.the	2.a	3.and	4.of	5.to	6.is	7.in	8.this	9.i	10.it
11.that	12.was	13.movie14.for	15.but	16.with	17.as	18.film	19.on	20.not
21.you	22.have	23.are	24.be	25.one	26.at	27.he	28.his	29.all	30.they
31.so	32.like	33.just	34.by	35.or	36.an	37.from	38.who	39.if	40.about
41.out	42.there43.it's	44.some	45.no	46.her	47.what	48.even	49.has	50.good
~~~
- 負面資料圖示
![image](https://github.com/williamchangTW/Data_mining_Project2/blob/master/%E8%9E%A2%E5%B9%95%E5%BF%AB%E7%85%A7%202018-12-03%20%E4%B8%8B%E5%8D%886.58.51.png)
- 包含正面及負面單字出現率排行 (前 50 名)
~~~
1.the	2.and	3.a	4.of	5.to	6.is	7.in	8.it	9.i	10.this
11.that	12.was	13.as	14.for	15.with	16.movie17.but	18.film	19.on	20.not
21.you	22.are	23.his	24.have	25.he	26.be	27.one	28.all	29.at	30.by
31.an	32.they	33.who	34.so	35.from	36.like	37.her	38.or	39.just	40.about
41.it's	42.out	43.if	44.has	45.some	46.there47.what	48.good	49.more	50.when
~~~ 
- 總和圖示
![image](https://github.com/williamchangTW/Data_mining_Project2/blob/master/%E8%9E%A2%E5%B9%95%E5%BF%AB%E7%85%A7%202018-12-03%20%E4%B8%8B%E5%8D%885.42.39.png)
- MLP 的建立，試用看看效果如何
~~~
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_2 (Embedding)      (None, 100, 32)           64000     
_________________________________________________________________
dropout_3 (Dropout)          (None, 100, 32)           0         
_________________________________________________________________
flatten_2 (Flatten)          (None, 3200)              0         
_________________________________________________________________
dense_3 (Dense)              (None, 256)               819456    
_________________________________________________________________
dropout_4 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 257       
=================================================================
Total params: 883,713
Trainable params: 883,713
Non-trainable params: 0
~~~

- 開始訓練
~~~
_________________________________________________________________

Train on 20000 samples, validate on 5000 samples
Epoch 1/10
 - 3s - loss: 0.0233 - acc: 0.9918 - val_loss: 1.5194 - val_acc: 0.7488
Epoch 2/10
 - 3s - loss: 0.0180 - acc: 0.9930 - val_loss: 1.2669 - val_acc: 0.7936
Epoch 3/10
 - 3s - loss: 0.0169 - acc: 0.9938 - val_loss: 1.6568 - val_acc: 0.7456
Epoch 4/10
 - 3s - loss: 0.0155 - acc: 0.9946 - val_loss: 1.6404 - val_acc: 0.7548
Epoch 5/10
 - 3s - loss: 0.0172 - acc: 0.9936 - val_loss: 1.8483 - val_acc: 0.7288
Epoch 6/10
 - 3s - loss: 0.0160 - acc: 0.9939 - val_loss: 1.8469 - val_acc: 0.7314
Epoch 7/10
 - 3s - loss: 0.0158 - acc: 0.9945 - val_loss: 1.4549 - val_acc: 0.7760
Epoch 8/10
 - 3s - loss: 0.0169 - acc: 0.9941 - val_loss: 1.9834 - val_acc: 0.7210
Epoch 9/10
 - 3s - loss: 0.0134 - acc: 0.9951 - val_loss: 2.0022 - val_acc: 0.7184
Epoch 10/10
 - 3s - loss: 0.0102 - acc: 0.9963 - val_loss: 2.1334 - val_acc: 0.7172
~~~
- 預測準確度
~~~
 25000/25000 [==============================] - 1s 53us/step
Out[78]:
0.81044
~~~
- 根據訓練結果建樹（1:正面，0:負面）
~~~
array([1, 1, 1, 0, 1, 1, 1, 1, 1, 1], dtype=int32)
~~~
- 預測真實資料
	- 預測是正面，結果是正面
~~~
Based on an actual story, John Boorman shows the struggle of an American doctor, whose husband and son were murdered and she was continually plagued with her loss. A holiday to Burma with her sister seemed like a good idea to get away from it all, but when her passport was stolen in Rangoon, she could not leave the country with her sister, and was forced to stay back until she could get I.D. papers from the American embassy. To fill in a day before she could fly out, she took a trip into the countryside with a tour guide. "I tried finding something in those stone statues, but nothing stirred in me. I was stone myself."   Suddenly all hell broke loose and she was caught in a political revolt. Just when it looked like she had escaped and safely boarded a train, she saw her tour guide get beaten and shot. In a split second she decided to jump from the moving train and try to rescue him, with no thought of herself. Continually her life was in danger.   Here is a woman who demonstrated spontaneous, selfless charity, risking her life to save another. Patricia Arquette is beautiful, and not just to look at; she has a beautiful heart. This is an unforgettable story.   "We are taught that suffering is the one promise that life always keeps."
('True value:', 'Positive', 'Prediction of result:', 'Positive')
~~~
- 預測真實資料
	- 預測為正面，但結果為負面
~~~
I have seen this movie and I did not care for this movie anyhow. I would not think about going to Paris because I do not like this country and its national capital. I do not like to learn french anyhow because I do not understand their language. Why would I go to France when I rather go to Germany or the United Kingdom? Germany and the United Kingdom are the nations I tolerate. Apparently the Olsen Twins do not understand the French language just like me. Therefore I will not bother the France trip no matter what. I might as well stick to the United Kingdom and meet single women and play video games if there is a video arcade. That is all.
('True value:', 'Negative', 'Prediction of result:', 'Positive')
~~~
- MLP 模型建立
~~~
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_4 (Embedding)      (None, 380, 32)           121600    
_________________________________________________________________
dropout_7 (Dropout)          (None, 380, 32)           0         
_________________________________________________________________
flatten_4 (Flatten)          (None, 12160)             0         
_________________________________________________________________
dense_6 (Dense)              (None, 256)               3113216   
_________________________________________________________________
dropout_8 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_7 (Dense)              (None, 1)                 257       
=================================================================
Total params: 3,235,073
Trainable params: 3,235,073
Non-trainable params: 0
_________________________________________________________________
~~~

- 開始訓練模型
~~~
Train on 20000 samples, validate on 5000 samples
Epoch 1/10
 - 8s - loss: 0.0134 - acc: 0.9965 - val_loss: 0.9137 - val_acc: 0.8128
Epoch 2/10
 - 7s - loss: 0.0095 - acc: 0.9975 - val_loss: 1.0084 - val_acc: 0.8062
Epoch 3/10
 - 7s - loss: 0.0101 - acc: 0.9969 - val_loss: 1.1539 - val_acc: 0.7916
Epoch 4/10
 - 7s - loss: 0.0085 - acc: 0.9978 - val_loss: 1.0509 - val_acc: 0.8170
Epoch 5/10
 - 7s - loss: 0.0119 - acc: 0.9956 - val_loss: 1.4173 - val_acc: 0.7690
Epoch 6/10
 - 7s - loss: 0.0092 - acc: 0.9964 - val_loss: 1.4380 - val_acc: 0.7770
Epoch 7/10
 - 7s - loss: 0.0086 - acc: 0.9976 - val_loss: 1.2963 - val_acc: 0.8014
Epoch 8/10
 - 7s - loss: 0.0087 - acc: 0.9973 - val_loss: 1.0945 - val_acc: 0.8242
Epoch 9/10
 - 7s - loss: 0.0096 - acc: 0.9967 - val_loss: 1.5664 - val_acc: 0.7692
Epoch 10/10
 - 7s - loss: 0.0073 - acc: 0.9977 - val_loss: 1.4521 - val_acc: 0.7828
~~~

- 顯示 MLP 模型變換預測結果
~~~
25000/25000 [==============================] - 3s 133us/step
Out[105]:
0.84188
~~~
- 這裡想用遞迴神經網路來幫忙解決問題，因為還不夠精確可能是只單純因單字去判斷，是否正面或負面。而很在看文章或是聽聲胤辨認這些行為時，可能是需要經由上下文才睛鞥準的判斷整段文章的意思，因此想要藉由 RNN 與 LSTM 來解決這樣的問題。 
	- RNN 模型測試
~~~
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_14 (Embedding)     (None, 380, 32)           121600    
_________________________________________________________________
dropout_27 (Dropout)         (None, 380, 32)           0         
_________________________________________________________________
simple_rnn_10 (SimpleRNN)    (None, 16)                784       
_________________________________________________________________
dense_26 (Dense)             (None, 256)               4352      
_________________________________________________________________
dropout_28 (Dropout)         (None, 256)               0         
_________________________________________________________________
dense_27 (Dense)             (None, 1)                 257       
=================================================================
Total params: 126,993
Trainable params: 126,993
Non-trainable params: 0
_________________________________________________________________
~~~

- 開始訓練
~~~
Train on 20000 samples, validate on 5000 samples
Epoch 1/10
 - 11s - loss: 0.6576 - acc: 0.6224 - val_loss: 0.8603 - val_acc: 4.0000e-04
Epoch 2/10
 - 10s - loss: 0.5040 - acc: 0.7570 - val_loss: 0.6052 - val_acc: 0.6914
Epoch 3/10
 - 11s - loss: 0.3602 - acc: 0.8470 - val_loss: 0.5943 - val_acc: 0.7436
Epoch 4/10
 - 10s - loss: 0.3047 - acc: 0.8770 - val_loss: 0.5665 - val_acc: 0.7644
Epoch 5/10
 - 10s - loss: 0.2757 - acc: 0.8916 - val_loss: 0.6411 - val_acc: 0.7424
Epoch 6/10
 - 10s - loss: 0.2529 - acc: 0.9022 - val_loss: 0.6686 - val_acc: 0.7482
Epoch 7/10
 - 10s - loss: 0.2323 - acc: 0.9102 - val_loss: 0.6393 - val_acc: 0.7720
Epoch 8/10
 - 10s - loss: 0.2141 - acc: 0.9170 - val_loss: 0.6136 - val_acc: 0.7784
Epoch 9/10
 - 10s - loss: 0.1906 - acc: 0.9295 - val_loss: 0.8541 - val_acc: 0.7224
Epoch 10/10
 - 11s - loss: 0.1781 - acc: 0.9317 - val_loss: 0.7273 - val_acc: 0.7674
~~~

- 顯示預測準確度
~~~
 25000/25000 [==============================] - 20s 801us/step
Out[140]:
0.8306
~~~

- LSTM 模型測試
~~~
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_15 (Embedding)     (None, 380, 32)           121600    
_________________________________________________________________
dropout_29 (Dropout)         (None, 380, 32)           0         
_________________________________________________________________
lstm_1 (LSTM)                (None, 32)                8320      
_________________________________________________________________
dense_28 (Dense)             (None, 256)               8448      
_________________________________________________________________
dropout_30 (Dropout)         (None, 256)               0         
_________________________________________________________________
dense_29 (Dense)             (None, 1)                 257       
=================================================================
Total params: 138,625
Trainable params: 138,625
Non-trainable params: 0
_________________________________________________________________
~~~

- 開始訓練模型
~~~
Train on 20000 samples, validate on 5000 samples
Epoch 1/10
 - 26s - loss: 0.2893 - acc: 0.8805 - val_loss: 0.6697 - val_acc: 0.7066
Epoch 2/10
 - 24s - loss: 0.2537 - acc: 0.8983 - val_loss: 0.4430 - val_acc: 0.8158
Epoch 3/10
 - 24s - loss: 0.2414 - acc: 0.9046 - val_loss: 0.5756 - val_acc: 0.7668
Epoch 4/10
 - 23s - loss: 0.2372 - acc: 0.9079 - val_loss: 0.4655 - val_acc: 0.8130
Epoch 5/10
 - 23s - loss: 0.2223 - acc: 0.9136 - val_loss: 0.5062 - val_acc: 0.8000
Epoch 6/10
 - 24s - loss: 0.2171 - acc: 0.9166 - val_loss: 0.4859 - val_acc: 0.8010
Epoch 7/10
 - 23s - loss: 0.2144 - acc: 0.9165 - val_loss: 0.5004 - val_acc: 0.8050
Epoch 8/10
 - 23s - loss: 0.2103 - acc: 0.9199 - val_loss: 0.4607 - val_acc: 0.8172
Epoch 9/10
 - 23s - loss: 0.1993 - acc: 0.9230 - val_loss: 0.4519 - val_acc: 0.8196
Epoch 10/10
 - 24s - loss: 0.1942 - acc: 0.9256 - val_loss: 0.4649 - val_acc: 0.8174
~~~

- 顯示預測結果
~~~
 25000/25000 [==============================] - 40s 2ms/step
Out[143]:
0.86592
~~~
- 為了解決評論字數太小預測錯誤，所以想用 ＬＳＴＭ 來進行訓練預測更加精準，因為包括長短程記憶能幫助上下文相關的預測，比如說遇到負負得正的詞彙能以正面的
 預測結果呈現。雖然精準度在測試這麼多模型後只上升了 0.5 左右（比起 MLP），應該還是有問題產生無法更加精準。
