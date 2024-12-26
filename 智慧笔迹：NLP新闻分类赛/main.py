import pandas as pd
from matplotlib import pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# 读取训练数据
# 数据读取： •使用pandas库读取训练数据集train_set.csv，
# 假设数据是以制表符\t分隔的，并且只读取前50000行以节省内存和处理时间。

train_df = pd.read_csv('train_set.csv', sep='\t', nrows=50000)


# 数据可视化：统计文本长度
# 数据可视化： •文本长度统计：计算每个文本（通过空格分割）的长度，并绘制直方图来可视化文本长度的分布。
#               这有助于理解数据的特性，如是否存在异常长的文本等。\
#            •类别分布：绘制类别（label）的条形图，以可视化不同类别的数量分布。这有助于了解数据集的类别平衡情况。

train_df['text_len'] = train_df['text'].apply(lambda x: len(x.split(' ')))
print(train_df['text_len'].describe())

plt.figure(figsize=(10, 6))
plt.hist(train_df['text_len'], bins=200)
plt.xlabel('Text char count')
plt.title("Histogram of char count")
plt.show()

# 数据可视化：类别分布
plt.figure(figsize=(10, 6))
train_df['label'].value_counts().plot(kind='bar')
plt.title('News class count')
plt.xlabel("Category")
plt.ylabel("Count")
plt.show()

# 统计词频
all_lines = ' '.join(list(train_df['text']))
word_count = Counter(all_lines.split(" "))
word_count = sorted(word_count.items(), key=lambda d: d[1], reverse=True)

print("总词数:", len(word_count))
print("出现频率最高的词:", word_count[0])
print("出现频率最低的词:", word_count[-1])

# 提取唯一字符的词频
# 词频统计： •统计所有文本中每个词的出现频率，并打印总词数、出现频率最高的词和最低的词。
#          •进一步地，对每个文本去重后（即每个文本只保留唯一的词），统计唯一词的频率，并打印相关信息。这有助于理解词汇的多样性。

train_df['text_unique'] = train_df['text'].apply(lambda x: ' '.join(list(set(x.split(' ')))))
all_lines_unique = ' '.join(list(train_df['text_unique']))
word_count_unique = Counter(all_lines_unique.split(" "))
word_count_unique = sorted(word_count_unique.items(), key=lambda d: d[1], reverse=True)

print("唯一词汇总数:", len(word_count_unique))
print("出现频率最高的唯一词:", word_count_unique[0])

# 特征提取：TF-IDF
# 特征提取： •使用TfidfVectorizer从文本中提取TF-IDF特征。
#           这里考虑了单词（ngram_range=(1, 1)的效果）和它们的二元、三元组合（ngram_range=(1, 3)），
#           并限制最多使用5000个特征。stop_words=None表示不自动去除停用词。

tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=5000, stop_words=None)
X = tfidf.fit_transform(train_df['text'])
y = train_df['label']

# 模型训练与验证：
# •将数据集分为训练集和验证集（80%训练，20%验证）。
# •使用RidgeClassifier（岭分类器）作为模型进行训练。岭分类器是一种线性分类器，通过引入L2正则化来处理过拟合问题。
# •在验证集上评估模型的性能，使用宏平均F1分数作为评估指标。宏平均F1分数是对每个类别分别计算F1分数，
# 然后取平均，不考虑类别的支持度（即类别样本数）。

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 模型训练
clf = RidgeClassifier()
clf.fit(X_train, y_train)

# 验证集评估
y_val_pred = clf.predict(X_val)
f1 = f1_score(y_val, y_val_pred, average='macro')
print(f"Validation F1 Score: {f1:.4f}")


# 预测与结果保存： •读取测试数据集test_a.csv，使用之前训练的模型和提取特征的TF-IDF向量化器对测试数据进行预测。
#               •将预测结果保存到hsl.csv文件中。
# 预测测试集
test_df = pd.read_csv('test_a.csv', sep='\t', nrows=50000)
X_test = tfidf.transform(test_df['text'])
y_test_pred = clf.predict(X_test)

# 保存预测结果
predictions_df = pd.DataFrame(y_test_pred, columns=['label'])
predictions_df.to_csv('hsl.csv', index=False)


