import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split




# 导入sklearn中的数据集
from sklearn import datasets


iris = datasets.load_iris() # 导入数据集
X = iris.data # 获得其特征向量
y = iris.target # 获得样本label
print('1.特征向量')
print(X)

print('2.样本label')
print(y)


# 拆分成训练集和验证集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



# print('X_tr shape:',.shape)


# 对测试集与验证集的所有推文进行BERT嵌入：
from bert_serving.client import BertClient

from sklearn.metrics import accuracy_score


# 导入训练数据
train = pd.read_csv('venv/c3_test3.csv', encoding='iso-8859-1')
train.shape




# 划分训练集与验证集
# X_tr, X_val, y_tr, y_val = train_test_split(train.clean_text, train.label, test_size=0.25, random_state=42)
X_tr, X_val, y_tr, y_val = train_test_split(train, train, test_size=0.25, random_state=42)
print('1.划分训练集和测试集')

print('X_tr shape:',X_tr.shape)

def clean_text(text):
    text = text

# 使用IP连接BERT服务器，目前bert服务器一直连不上，错误日志在wget-log

bc = BertClient(ip="YOUR_SERVER_IP")
print(bc)
# 获得训练集与测试集的嵌入

X_tr_bert = bc.encode(X_tr.tolist())
X_val_bert = bc.encode(X_val.tolist())

# 构建模型了！先来训练分类模型：

from sklearn.linear_model import LogisticRegression

# LR模型
model_bert = LogisticRegression()
# 训练
model_bert = model_bert.fit(X_tr_bert, y_tr)
# 预测
pred_bert = model_bert.predict(X_val_bert)

# 查看分类准确率：
print('2.查看分类准确率')

print(accuracy_score(y_val, pred_bert))
