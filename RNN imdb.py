import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Embedding, SimpleRNN
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.datasets import imdb

# Load the IMDb dataset
max_words = 5000
max_len = 600
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_words)

# Pad sequences to the same length
X_train = sequence.pad_sequences(X_train, maxlen=max_len)
X_test = sequence.pad_sequences(X_test, maxlen=max_len)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

# Define RNN model
inputs = Input(name='inputs', shape=[max_len])
layer = Embedding(max_words + 1, 128, input_length=max_len)(inputs)
layer = SimpleRNN(128)(layer)
layer = Dense(128, activation="relu", name="FC1")(layer)
layer = Dropout(0.5)(layer)
layer = Dense(2, activation="softmax", name="FC2")(layer)  # Assuming binary classification
model = Model(inputs=inputs, outputs=layer)
model.summary()
model.compile(loss="categorical_crossentropy", optimizer="RMSprop", metrics=["accuracy"])

# Convert labels to one-hot encoding
y_train_one_hot = tf.keras.utils.to_categorical(y_train, 2)  # Assuming binary classification
y_val_one_hot = tf.keras.utils.to_categorical(y_val, 2)
y_test_one_hot = tf.keras.utils.to_categorical(y_test, 2)

# Train the model
history=model.fit(X_train, y_train_one_hot, batch_size=128, epochs=30,
          validation_data=(X_val, y_val_one_hot),
          callbacks=[TensorBoard(log_dir='./log')])

# Save the model
model.save('./model/RNN_imdb.h5')
del model

# Load the pre-trained model
model = tf.keras.models.load_model('./model/RNN_imdb.h5')

# Predict on the test set
test_pre = model.predict(X_test)
pred = np.argmax(test_pre, axis=1)
real = y_test
cv_conf = confusion_matrix(real, pred)
acc = accuracy_score(real, pred)
precision = precision_score(real, pred, average='micro')
recall = recall_score(real, pred, average='micro')
f1 = f1_score(real, pred, average='micro')
pattern = 'test:  acc: %.4f   precision: %.4f   recall: %.4f   f1: %.4f'
print(pattern % (acc, precision, recall, f1))

labels11 = ['negative', 'positive']  # Assuming binary classification
disp = ConfusionMatrixDisplay(confusion_matrix=cv_conf, display_labels=labels11)
disp.plot(cmap="Blues", values_format='')
plt.savefig("ConfusionMatrix2.png", dpi=400)
plt.clf()

# history = model.fit(X_train, y_train_one_hot, batch_size=128, epochs=30,
#                     validation_data=(X_val, y_val_one_hot),
#                     callbacks=[TensorBoard(log_dir='./log')])

# Plot accuracy
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Validation'], loc='upper left')
plt.savefig('AccuracyCurve2.png', dpi=400)
plt.clf()


# Plot loss
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Validation'], loc='upper left')
plt.savefig('LossCurve2.png', dpi=400)
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
plt.savefig('ROC_curve2.png', dpi=300)
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


#imdb
#30
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

# 137/137 [==============================] - 20s 139ms/step - loss: 0.7166 - accuracy: 0.5175 - val_loss: 0.6713 - val_accuracy: 0.5703
# Epoch 2/30
# 137/137 [==============================] - 18s 134ms/step - loss: 0.5852 - accuracy: 0.6898 - val_loss: 1.2373 - val_accuracy: 0.5576
# Epoch 3/30
# 137/137 [==============================] - 18s 135ms/step - loss: 0.4786 - accuracy: 0.7867 - val_loss: 0.4250 - val_accuracy: 0.8157
# Epoch 4/30
# 137/137 [==============================] - 18s 133ms/step - loss: 0.4141 - accuracy: 0.8236 - val_loss: 0.4469 - val_accuracy: 0.8155
# Epoch 5/30
# 137/137 [==============================] - 18s 133ms/step - loss: 0.3962 - accuracy: 0.8379 - val_loss: 0.4042 - val_accuracy: 0.8332
# Epoch 6/30
# 137/137 [==============================] - 18s 134ms/step - loss: 0.3732 - accuracy: 0.8488 - val_loss: 0.5693 - val_accuracy: 0.8027
# Epoch 7/30
# 137/137 [==============================] - 18s 133ms/step - loss: 0.4082 - accuracy: 0.8205 - val_loss: 0.4202 - val_accuracy: 0.8125
# Epoch 8/30
# 137/137 [==============================] - 18s 133ms/step - loss: 0.3800 - accuracy: 0.8429 - val_loss: 0.5234 - val_accuracy: 0.7369
# Epoch 9/30
# 137/137 [==============================] - 18s 132ms/step - loss: 0.3846 - accuracy: 0.8399 - val_loss: 0.4311 - val_accuracy: 0.8137
# Epoch 10/30
# 137/137 [==============================] - 18s 131ms/step - loss: 0.3633 - accuracy: 0.8505 - val_loss: 0.4621 - val_accuracy: 0.8231
# Epoch 11/30
# 137/137 [==============================] - 18s 131ms/step - loss: 0.3634 - accuracy: 0.8640 - val_loss: 0.4814 - val_accuracy: 0.8173
# Epoch 12/30
# 137/137 [==============================] - 18s 135ms/step - loss: 0.3238 - accuracy: 0.8718 - val_loss: 0.4377 - val_accuracy: 0.8276
# Epoch 13/30
# 137/137 [==============================] - 18s 132ms/step - loss: 0.2976 - accuracy: 0.8864 - val_loss: 0.4907 - val_accuracy: 0.8096
# Epoch 14/30
# 137/137 [==============================] - 19s 136ms/step - loss: 0.2858 - accuracy: 0.8871 - val_loss: 0.4438 - val_accuracy: 0.8109
# Epoch 15/30
# 137/137 [==============================] - 18s 133ms/step - loss: 0.2771 - accuracy: 0.8903 - val_loss: 0.5208 - val_accuracy: 0.8191
# Epoch 16/30
# 137/137 [==============================] - 18s 133ms/step - loss: 0.2472 - accuracy: 0.9088 - val_loss: 0.5522 - val_accuracy: 0.8232
# Epoch 17/30
# 137/137 [==============================] - 18s 132ms/step - loss: 0.2340 - accuracy: 0.9110 - val_loss: 0.4882 - val_accuracy: 0.8344
# Epoch 18/30
# 137/137 [==============================] - 18s 133ms/step - loss: 0.2673 - accuracy: 0.8942 - val_loss: 0.5582 - val_accuracy: 0.7920
# Epoch 19/30
# 137/137 [==============================] - 18s 134ms/step - loss: 0.2356 - accuracy: 0.9083 - val_loss: 0.4805 - val_accuracy: 0.8163
# Epoch 20/30
# 137/137 [==============================] - 18s 133ms/step - loss: 0.2423 - accuracy: 0.9074 - val_loss: 0.6224 - val_accuracy: 0.7347
# Epoch 21/30
# 137/137 [==============================] - 18s 133ms/step - loss: 0.2097 - accuracy: 0.9208 - val_loss: 0.5517 - val_accuracy: 0.7880
# Epoch 22/30
# 137/137 [==============================] - 18s 134ms/step - loss: 0.1821 - accuracy: 0.9348 - val_loss: 0.5963 - val_accuracy: 0.7944
# Epoch 23/30
# 137/137 [==============================] - 18s 132ms/step - loss: 0.1527 - accuracy: 0.9450 - val_loss: 0.6436 - val_accuracy: 0.7940
# Epoch 24/30
# 137/137 [==============================] - 18s 133ms/step - loss: 0.1283 - accuracy: 0.9539 - val_loss: 0.7090 - val_accuracy: 0.7815
# Epoch 25/30
# 137/137 [==============================] - 18s 133ms/step - loss: 0.1034 - accuracy: 0.9635 - val_loss: 0.9643 - val_accuracy: 0.7657
# Epoch 26/30
# 137/137 [==============================] - 19s 140ms/step - loss: 0.0965 - accuracy: 0.9661 - val_loss: 0.8409 - val_accuracy: 0.7956
# Epoch 27/30
# 137/137 [==============================] - 18s 135ms/step - loss: 0.0819 - accuracy: 0.9724 - val_loss: 0.8986 - val_accuracy: 0.7517
# Epoch 28/30
# 137/137 [==============================] - 19s 136ms/step - loss: 0.0645 - accuracy: 0.9789 - val_loss: 0.8562 - val_accuracy: 0.7944
# Epoch 29/30
# 137/137 [==============================] - 18s 134ms/step - loss: 0.0633 - accuracy: 0.9813 - val_loss: 1.2125 - val_accuracy: 0.8015
# Epoch 30/30
# 137/137 [==============================] - 18s 134ms/step - loss: 0.0518 - accuracy: 0.9840 - val_loss: 1.1512 - val_accuracy: 0.7896
# C:\Users\26789\AppData\Local\Programs\Python\Python311\Lib\site-packages\keras\src\engine\training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
#   saving_api.save_model(
# 782/782 [==============================] - 13s 16ms/step
# test:  acc: 0.7868   precision: 0.7868   recall: 0.7868   f1: 0.7868
# K-S Statistic: 0.008367568254470825