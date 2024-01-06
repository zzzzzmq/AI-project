import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Embedding, SimpleRNN
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
import jieba
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score
from tensorflow.keras.callbacks import TensorBoard
import pandas as pd
from tensorflow.keras.datasets import imdb
import seaborn as sns
# 数据处理
def is_chinese(uchar):
    if '\u4e00' <= uchar <= '\u9fa5':
        return True
    return False

def reserve_chinese(content):
    content_str = ''
    for i in content:
        if is_chinese(i):
            content_str += i
    return content_str

def data_parse(text, stop_words):
    label, content = text.split('	####	')
    content = reserve_chinese(content)
    words = jieba.lcut(content)
    words = [i for i in words if i not in stop_words]
    return words, int(label)

def get_stop_words():
    file = open('./data/stopwords.txt', 'r', encoding='utf8')
    words = [i.strip() for i in file.readlines()]
    file.close()
    return words

def get_formatted_data():
    file = open('./data/dmsc.txt', 'r', encoding='utf8')
    texts = file.readlines()
    file.close()
    stop_words = get_stop_words()
    all_words = []
    all_labels = []
    count=0
    for text in texts:
        count=count+1
        if '####' in text and count<=149800:# 检查每行数据是否包含 "#"
            content, label = data_parse(text, stop_words)
            if len(content) <= 0:
                continue
            all_words.append(content)
            all_labels.append(label)
    return all_words, all_labels

# 读取数据集
data, label = get_formatted_data()
X_train, X_t, train_y, v_y = train_test_split(data, label, test_size=0.3, random_state=42)
X_val, X_test, val_y, test_y = train_test_split(X_t, v_y, test_size=0.3, random_state=42)

# 对标签数据进行编码
le = LabelEncoder()
train_y = le.fit_transform(train_y).reshape(-1, 1)
val_y = le.transform(val_y).reshape(-1, 1)
test_y = le.transform(test_y).reshape(-1, 1)

print(train_y.shape)
print(val_y.shape)
print(test_y.shape)

# 对标签数据进行one-hot编码
ohe = OneHotEncoder()
train_y = ohe.fit_transform(train_y).toarray()
val_y = ohe.transform(val_y).toarray()
test_y = ohe.transform(test_y).toarray()

# 使用Tokenizer对词组进行编码
max_words = 5000
max_len = 600
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(data)

train_seq = tok.texts_to_sequences(X_train)
val_seq = tok.texts_to_sequences(X_val)
test_seq = tok.texts_to_sequences(X_test)

# 将每个序列调整为相同的长度
train_seq_mat = sequence.pad_sequences(train_seq, maxlen=max_len)
val_seq_mat = sequence.pad_sequences(val_seq, maxlen=max_len)
test_seq_mat = sequence.pad_sequences(test_seq, maxlen=max_len)

# 定义RNN模型
inputs = Input(name='inputs', shape=[max_len])
layer = Embedding(max_words + 1, 128, input_length=max_len)(inputs)
layer = SimpleRNN(128)(layer)
layer = Dense(128, activation="relu", name="FC1")(layer)
layer = Dropout(0.5)(layer)
layer = Dense(2, activation="softmax", name="FC2")(layer)
model = Model(inputs=inputs, outputs=layer)
model.summary()
model.compile(loss="categorical_crossentropy", optimizer="RMSprop", metrics=["accuracy"])

# 模型训练
history = model.fit(train_seq_mat, train_y, batch_size=128, epochs=30,
          validation_data=(val_seq_mat, val_y),
          callbacks=[TensorBoard(log_dir='./log')])

# Store accuracy and loss values
train_acc_values = history.history['accuracy']
val_acc_values = history.history['val_accuracy']
train_loss_values = history.history['loss']
val_loss_values = history.history['val_loss']


# 保存模型
model.save('./model/RNN.h5')
del model

# 导入已经训练好的模型
model = tf.keras.models.load_model('./model/RNN.h5')

# 对测试集进行预测
test_pre = model.predict(test_seq_mat)
pred = np.argmax(test_pre, axis=1)
real = np.argmax(test_y, axis=1)
cv_conf = confusion_matrix(real, pred)
acc = accuracy_score(real, pred)
precision = precision_score(real, pred, average='micro')
recall = recall_score(real, pred, average='micro')
f1 = f1_score(real, pred, average='micro')
pattern = 'test:  acc: %.4f   precision: %.4f   recall: %.4f   f1: %.4f'
print(pattern % (acc, precision, recall, f1))

labels11 = ['negative', 'active']
disp = ConfusionMatrixDisplay(confusion_matrix=cv_conf, display_labels=labels11)
disp.plot(cmap="Blues", values_format='')
plt.savefig("ConfusionMatrix1.png", dpi=400)
plt.clf()

# Plot accuracy
plt.plot(val_acc_values)
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Validation'], loc='upper left')
plt.savefig('AccuracyCurve1.png', dpi=400)
plt.clf()


# Plot loss
plt.plot(val_loss_values)
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Validation'], loc='upper left')
plt.savefig('LossCurve1.png', dpi=400)
plt.clf()

from sklearn.metrics import roc_curve, roc_auc_score

# 计算每个类别的预测概率
positive_class_index = 1  # 正类的索引
# 提取正类的预测概率
positive_probs = test_pre[:, positive_class_index]
# 计算真正率（True Positive Rate）和假正率（False Positive Rate）
fpr, tpr, thresholds = roc_curve(real, positive_probs)

# 计算 AUC（Area Under the Curve）值
auc = roc_auc_score(real, positive_probs)

# 绘制 ROC 曲线
plt.plot(fpr, tpr, label='ROC curve (area = %.2f)' % auc)
plt.plot([0, 1], [0, 1], 'k--')  # 对角线
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig('ROC_curve1.png', dpi=300)
plt.clf()

# 计算每个类别的预测概率
positive_class_index = 1  # 正类的索引，这里假设正类的索引为1
# 提取正类的预测概率
positive_probs = test_pre[:, positive_class_index]
# 计算负类的预测概率
negative_probs = test_pre[:, 1 - positive_class_index]

# 计算正类和负类的累积分布函数（CDF）
positive_cdf = np.cumsum(positive_probs) / np.sum(positive_probs)
negative_cdf = np.cumsum(negative_probs) / np.sum(negative_probs)

# 计算K-S统计量
ks_statistic = np.max(np.abs(positive_cdf - negative_cdf))

# 输出K-S统计量
print(f'K-S Statistic: {ks_statistic}')
# K-S Statistic: 0.022289305925369263



#dmsc.txt
# Model: "model"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  inputs (InputLayer)         [(None, 600)]             0

#  embedding (Embedding)       (None, 600, 128)          640128

#  simple_rnn (SimpleRNN)      (None, 128)               32896

#  FC1 (Dense)                 (None, 128)               16512

#  dropout (Dropout)           (None, 128)               0

#  FC2 (Dense)                 (None, 2)                 258

# =================================================================
# Total params: 689794 (2.63 MB)
# Trainable params: 689794 (2.63 MB)
# Non-trainable params: 0 (0.00 Byte)
# _________________________________________________________________
# WARNING:tensorflow:From C:\Users\26789\AppData\Local\Programs\Python\Python311\Lib\site-packages\keras\src\optimizers\__init__.py:309: The name tf.train.Optimizer is deprecated. Please use tf.compat.v1.train.Optimizer instead.

# Epoch 1/30
# WARNING:tensorflow:From C:\Users\26789\AppData\Local\Programs\Python\Python311\Lib\site-packages\keras\src\utils\tf_utils.py:492: The name tf.ragged.RaggedTensorValue is deprecated. Please use tf.compat.v1.ragged.RaggedTensorValue instead.

# WARNING:tensorflow:From C:\Users\26789\AppData\Local\Programs\Python\Python311\Lib\site-packages\keras\src\engine\base_layer_utils.py:384: The name tf.executing_eagerly_outside_functions is deprecated. Please use tf.compat.v1.executing_eagerly_outside_functions instead.

# 806/806 [==============================] - 105s 129ms/step - loss: 0.4299 - accuracy: 0.8142 - val_loss: 0.3541 - val_accuracy: 0.8505
# Epoch 2/30
# 806/806 [==============================] - 103s 128ms/step - loss: 0.3739 - accuracy: 0.8434 - val_loss: 0.3530 - val_accuracy: 0.8560
# Epoch 3/30
# 806/806 [==============================] - 104s 130ms/step - loss: 0.3387 - accuracy: 0.8614 - val_loss: 0.3411 - val_accuracy: 0.8617
# Epoch 4/30
# 806/806 [==============================] - 103s 128ms/step - loss: 0.3266 - accuracy: 0.8678 - val_loss: 0.3322 - val_accuracy: 0.8617
# Epoch 5/30
# 806/806 [==============================] - 105s 130ms/step - loss: 0.3144 - accuracy: 0.8728 - val_loss: 0.3266 - val_accuracy: 0.8633
# Epoch 6/30
# 806/806 [==============================] - 103s 128ms/step - loss: 0.3074 - accuracy: 0.8763 - val_loss: 0.3398 - val_accuracy: 0.8637
# Epoch 7/30
# 806/806 [==============================] - 106s 132ms/step - loss: 0.3018 - accuracy: 0.8792 - val_loss: 0.3258 - val_accuracy: 0.8649
# Epoch 8/30
# 806/806 [==============================] - 104s 128ms/step - loss: 0.2963 - accuracy: 0.8805 - val_loss: 0.3866 - val_accuracy: 0.8591
# Epoch 9/30
# 806/806 [==============================] - 104s 129ms/step - loss: 0.2905 - accuracy: 0.8831 - val_loss: 0.3309 - val_accuracy: 0.8636
# Epoch 10/30
# 806/806 [==============================] - 104s 129ms/step - loss: 0.2857 - accuracy: 0.8849 - val_loss: 0.3434 - val_accuracy: 0.8616
# Epoch 11/30
# 806/806 [==============================] - 103s 128ms/step - loss: 0.2813 - accuracy: 0.8875 - val_loss: 0.3445 - val_accuracy: 0.8549
# Epoch 12/30
# 806/806 [==============================] - 104s 129ms/step - loss: 0.2738 - accuracy: 0.8902 - val_loss: 0.3432 - val_accuracy: 0.8626
# Epoch 13/30
# 806/806 [==============================] - 103s 128ms/step - loss: 0.2702 - accuracy: 0.8928 - val_loss: 0.3511 - val_accuracy: 0.8628
# Epoch 14/30
# 806/806 [==============================] - 103s 128ms/step - loss: 0.2625 - accuracy: 0.8958 - val_loss: 0.3478 - val_accuracy: 0.8606
# Epoch 15/30
# 806/806 [==============================] - 103s 128ms/step - loss: 0.2571 - accuracy: 0.8988 - val_loss: 0.3599 - val_accuracy: 0.8608
# Epoch 16/30
# 806/806 [==============================] - 103s 128ms/step - loss: 0.2508 - accuracy: 0.9025 - val_loss: 0.3658 - val_accuracy: 0.8455
# Epoch 17/30
# 806/806 [==============================] - 104s 129ms/step - loss: 0.2457 - accuracy: 0.9042 - val_loss: 0.3794 - val_accuracy: 0.8601
# Epoch 18/30
# 806/806 [==============================] - 104s 129ms/step - loss: 0.2436 - accuracy: 0.9049 - val_loss: 0.3788 - val_accuracy: 0.8512
# Epoch 19/30
# 806/806 [==============================] - 104s 128ms/step - loss: 0.2373 - accuracy: 0.9081 - val_loss: 0.3727 - val_accuracy: 0.8522
# Epoch 20/30
# 806/806 [==============================] - 103s 128ms/step - loss: 0.2304 - accuracy: 0.9104 - val_loss: 0.3955 - val_accuracy: 0.8360
# Epoch 21/30
# 806/806 [==============================] - 103s 128ms/step - loss: 0.2210 - accuracy: 0.9153 - val_loss: 0.3974 - val_accuracy: 0.8481
# Epoch 22/30
# 806/806 [==============================] - 104s 129ms/step - loss: 0.2152 - accuracy: 0.9173 - val_loss: 0.4188 - val_accuracy: 0.8502
# Epoch 23/30
# 806/806 [==============================] - 104s 129ms/step - loss: 0.2092 - accuracy: 0.9203 - val_loss: 0.4061 - val_accuracy: 0.8478
# Epoch 24/30
# 806/806 [==============================] - 102s 127ms/step - loss: 0.2073 - accuracy: 0.9221 - val_loss: 0.4389 - val_accuracy: 0.8462
# Epoch 25/30
# 806/806 [==============================] - 103s 128ms/step - loss: 0.2012 - accuracy: 0.9235 - val_loss: 0.4289 - val_accuracy: 0.8462
# Epoch 26/30
# 806/806 [==============================] - 103s 128ms/step - loss: 0.1948 - accuracy: 0.9267 - val_loss: 0.4433 - val_accuracy: 0.8508
# Epoch 27/30
# 806/806 [==============================] - 103s 127ms/step - loss: 0.1885 - accuracy: 0.9293 - val_loss: 0.4544 - val_accuracy: 0.8429
# Epoch 28/30
# 806/806 [==============================] - 103s 128ms/step - loss: 0.1819 - accuracy: 0.9319 - val_loss: 0.4620 - val_accuracy: 0.8357
# Epoch 29/30
# 806/806 [==============================] - 103s 127ms/step - loss: 0.1811 - accuracy: 0.9322 - val_loss: 0.4896 - val_accuracy: 0.8417
# Epoch 30/30
# 806/806 [==============================] - 103s 128ms/step - loss: 0.1690 - accuracy: 0.9376 - val_loss: 0.5242 - val_accuracy: 0.8339
# C:\Users\26789\AppData\Local\Programs\Python\Python311\Lib\site-packages\keras\src\engine\training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
#   saving_api.save_model(
# 415/415 [==============================] - 7s 16ms/step
# test:  acc: 0.8311   precision: 0.8311   recall: 0.8311   f1: 0.8311
# K-S Statistic: 0.013724029064178467