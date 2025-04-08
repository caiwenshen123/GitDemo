### 代码结构和功能

1. **导入必要的库**
   ```python
   import re
   import os
   from jieba import cut
   from itertools import chain
   from collections import Counter
   import numpy as np
   from sklearn.naive_bayes import MultinomialNB
   ```
   - `re`：用于正则表达式处理。
   - `os`：用于文件路径和操作。
   - `jieba`：中文分词库。
   - `itertools.chain`：用于连接多个列表。
   - `collections.Counter`：用于统计词频。
   - `numpy`：用于数组操作。
   - `sklearn.naive_bayes.MultinomialNB`：用于构建朴素贝叶斯分类器。

2. **获取和处理文本数据**
   ```python
   def get_words(filename):
       """读取文本并过滤无效字符和长度为1的词"""
       words = []
       with open(filename, 'r', encoding='utf-8') as fr:
           for line in fr:
               line = line.strip()
               # 过滤无效字符
               line = re.sub(r'[.【】0-9、——。，！~\*]', '', line)
               # 使用jieba.cut()方法对文本切词处理
               line = cut(line)
               # 过滤长度为1的词
               line = filter(lambda word: len(word) > 1, line)
               words.extend(line)
       return words
   ```
   - 该函数从指定文件中读取文本，去掉无效字符（如标点和数字），用 `jieba` 库进行中文分词，最终返回一个词的列表。

3. **建立词库**
   ```python
   all_words = []
   def get_top_words(top_num):
       """遍历邮件建立词库后返回出现次数最多的词"""
       filename_list = ['邮件_files/{}.txt'.format(i) for i in range(151)]
       for filename in filename_list:
           all_words.append(get_words(filename))
       freq = Counter(chain(*all_words))
       return [i[0] for i in freq.most_common(top_num)]
   ```
   - `get_top_words()` 函数读取多个邮件文件，统计每个词出现的频率，并返回出现次数最多的 `top_num` 个词。

4. **构建特征向量**
   ```python
   top_words = get_top_words(100)
   vector = []
   for words in all_words:
       word_map = list(map(lambda word: words.count(word), top_words))
       vector.append(word_map)
   vector = np.array(vector)
   ```
   - 将读取到的所有邮件的词进行处理，构建特征向量。每个邮件对应一个向量，向量的每个元素表示该邮件中某个词的出现次数。

5. **准备标签**
   ```python
   labels = np.array([1]*127 + [0]*24)
   ```
   - 为邮件设置标签：0 表示普通邮件，1 表示垃圾邮件。前 127 个邮件被标记为垃圾邮件，后 24 个邮件标记为普通邮件。

6. **训练模型**
   ```python
   model = MultinomialNB()
   model.fit(vector, labels)
   ```
   - 利用训练数据（特征向量和标签）训练朴素贝叶斯模型。

7. **预测新邮件类型**
   ```python
   def predict(filename):
       """对未知邮件分类"""
       words = get_words(filename)
       current_vector = np.array(
           tuple(map(lambda word: words.count(word), top_words)))
       result = model.predict(current_vector.reshape(1, -1))
       return '垃圾邮件' if result == 1 else '普通邮件'
   ```
   - `predict()` 函数用于对新的邮件进行分类。它首先读取邮件并获取词向量，然后使用训练好的模型进行预测。根据预测结果返回“垃圾邮件”或“普通邮件”。

8. **测试和输出分类结果**
   ```python
   print('151.txt分类情况:{}'.format(predict('邮件_files/151.txt')))
   print('152.txt分类情况:{}'.format(predict('邮件_files/152.txt')))
   print('153.txt分类情况:{}'.format(predict('邮件_files/153.txt')))
   print('154.txt分类情况:{}'.format(predict('邮件_files/154.txt')))
   print('155.txt分类情况:{}'.format(predict('邮件_files/155.txt')))
   ```
   - 这些打印语句用于测试整个流程，通过调用 `predict()` 函数来分类特定的邮件文件。
   - 
<img src="https://github.com/caiwenshen123/GitDemo/blob/master/images/p1.png" width="800" alt="截图一">