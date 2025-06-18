# 邮件分类系统

基于朴素贝叶斯算法的邮件分类系统，支持高频词和TF-IDF两种特征提取模式，并包含数据平衡功能。

## 功能特点

- 文本预处理：清洗字符、分词、过滤单字词
- 双特征模式：高频词(Word Frequency)和TF-IDF
- 数据平衡：使用SMOTE算法处理样本不平衡问题
- 分类模型：多项式朴素贝叶斯(MultinomialNB)
- 模型评估：提供详细的分类性能报告

## 核心功能说明

### 1. 获取邮件内容并切词处理

```python
def get_words(filename):
    words = []
    with open(filename, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.strip()
            line = re.sub(r'[.【】0-9、——。，！~\*]', '', line)  # 过滤无效字符
            line = cut(line)  # jieba分词
            line = filter(lambda word: len(word) > 1, line)  # 过滤长度为1的词
            words.extend(line)
    return words
```

该函数读取邮件文本文件，进行以下处理：
- 去除标点符号、数字等无效字符
- 使用jieba进行中文分词
- 过滤长度为1的单字词
- 返回处理后的词列表

### 2. 构建词汇表并提取频率较高的词

```python
def get_top_words(top_num):
    filename_list = ['邮件_files/{}.txt'.format(i) for i in range(151)]
    for filename in filename_list:
        all_words.append(get_words(filename))  # 遍历所有邮件生成词库
    freq = Counter(chain(*all_words))  # 统计词频
    return [i[0] for i in freq.most_common(top_num)]  # 返回前 top_num 个高频词
```

该函数用于构建词汇表：
- 遍历所有训练邮件
- 统计所有词的出现频率
- 返回出现频率最高的top_num个词作为特征词

### 3. 构建邮件的词向量

```python
vector = []
for words in all_words:
    word_map = list(map(lambda word: words.count(word), top_words))  # 统计每个特征词的词频
    vector.append(word_map)
vector = np.array(vector)  # 转换为 NumPy 数组
```

该部分将邮件转换为特征向量：
- 对于每封邮件，统计每个特征词的出现次数
- 将统计结果组成词频向量
- 转换为NumPy数组格式，便于模型处理

### 4. 构建分类模型

```python
model = MultinomialNB()  # 初始化多项式朴素贝叶斯模型
model.fit(vector, labels)  # 使用词频向量和标签进行训练
```

使用多项式朴素贝叶斯算法构建分类模型：
- 初始化MultinomialNB模型
- 使用特征向量和对应的标签进行训练
- 适合处理离散的词频特征

### 5. 对未知邮件进行分类

```python
def predict(filename):
    words = get_words(filename)  # 预处理新邮件
    current_vector = np.array(tuple(map(lambda word: words.count(word), top_words)))  # 生成词频向量
    result = model.predict(current_vector.reshape(1, -1))  # 预测结果
    return '垃圾邮件' if result == 1 else '普通邮件'
```

该函数用于对新邮件进行分类：
- 对新邮件进行相同的预处理
- 生成对应的特征向量
- 使用训练好的模型进行预测
- 返回分类结果（垃圾邮件或普通邮件）

### 6. 使用SMOTE进行数据平衡

```python
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)
```

当训练数据不平衡时（如垃圾邮件和普通邮件数量差异较大），使用SMOTE算法进行数据平衡：
- 自动识别少数类并生成合成样本
- 保持数据的分布特征
- 提高模型对少数类的识别能力

## 高频词/TF-IDF两种特征模式的切换方法

### 1. 高频词(Word Frequency)特征模式

高频词特征模式是指根据每个词在整个文本中的出现频率来表示文本。该方法简单直观，生成的特征向量通常是每个词在文本中出现的次数。

### 2. TF-IDF特征模式

TF-IDF是一种考虑到词频(Term Frequency)和逆文档频率(Inverse Document Frequency)的加权方法：
- TF：词在文档中的出现频率
- IDF：词在整个语料库中的稀有程度
- 该方法不仅反映了一个词在文档中的重要性，还考虑了词在整个语料库中的分布情况

### 3. 如何切换

可以通过调整文本特征提取的方式来在高频词和TF-IDF之间进行切换，使用sklearn提供的CountVectorizer和TfidfVectorizer来处理文本特征：

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# 假设我们有一个文本数据集
documents = [
    "This is a spam message",
    "This is a ham message",
    "Free money is waiting for you",
    "Win a lottery by sending your email",
]

# 高频词特征模式：使用CountVectorizer
count_vectorizer = CountVectorizer(stop_words='english')
X_count = count_vectorizer.fit_transform(documents)
print("High-frequency features:")
print(count_vectorizer.get_feature_names_out())  # 输出特征词汇

# TF-IDF特征模式：使用TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = tfidf_vectorizer.fit_transform(documents)
print("\nTF-IDF features:")
print(tfidf_vectorizer.get_feature_names_out())  # 输出特征词汇
```

## 使用方法

1. 准备数据：
   - 将邮件按编号1-151存放于`邮件_files`目录
   - 1-126为垃圾邮件，127-151为普通邮件
   - 152-155为测试邮件

2. 运行分类器：
   ```python
   # 高频词模式
   python email_classifier.py --mode word_freq
   
   # TF-IDF模式
   python email_classifier.py --mode tfidf
   ```

3. 查看分类结果和模型评估报告

## 示例运行结果
<img src="https://github.com/caiwenshen123/GitDemo/blob/master/images/test 4-1.png." width="800" alt="截图一">

## 样本平衡处理
<img src="https://github.com/caiwenshen123/GitDemo/blob/master/images/test 4-2.png." width="800" alt="截图二">

## 增加模型评估指标
<img src="https://github.com/caiwenshen123/GitDemo/blob/master/images/test 4-3.png." width="800" alt="截图三">