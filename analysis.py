# 导入必备工具包
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
# 设置显示风格
plt.style.use('fivethirtyeight') 


# 分别读取训练tsv和验证tsv
train_data = pd.read_csv("./data/ag_news_csv/train.tsv", sep="\t")
print(train_data.columns)


'''
获取标签数据分布
'''
plt.figure()
# 获得训练数据标签数量分布
sns.countplot("label", data=train_data)
plt.title("train_data")
plt.show()
plt.savefig("./image/label_count.png")

'''
合并标题和文本，生成新的列
'''
train_data["sentence"] = list(map(lambda x,y: x + " " + y, train_data["sentence1"], train_data["sentence2"]))

'''
获取句子长度分布
'''
# 在训练数据中添加新的句子长度列, 每个元素的值都是对应的句子列的长度
train_data["sentence_length"] = list(map(lambda x: len(x), train_data["sentence"]))

# 绘制句子长度列的数量分布图
plt.figure()
sns.countplot("sentence_length", data=train_data)
# 主要关注count长度分布的纵坐标, 不需要绘制横坐标, 横坐标范围通过dist图进行查看
plt.xticks([])
plt.show()
plt.savefig("./image/train_length_count.png")

# 绘制dist长度分布图
plt.figure()
sns.distplot(train_data["sentence_length"])
# 主要关注dist长度分布横坐标, 不需要绘制纵坐标
plt.yticks([])
plt.show()
plt.savefig("./image/train_length_dis.png")


'''
获取正负样本长度散点分布
'''
# 绘制训练集长度分布的散点图
plt.figure()
sns.stripplot(y='sentence_length',x='label',data=train_data)
plt.show()
plt.savefig("./image/train_length_scatter.png")


'''
获取词汇总数统计
'''
# 导入chain方法用于扁平化列表, 去除 iterable 里的内嵌 iterable
# a = [(1, 'a'), (2, 'b'), (3, 'c')]
# list(chain(*a)
# [1, 'a', 2, 'b', 3, 'c']
     
import jieba
from itertools import chain
# 进行训练集的句子进行分词, 并统计出不同词汇的总数
train_vocab = set(chain(*map(lambda x: jieba.lcut(x), train_data["sentence"])))
print("训练集共包含不同词汇总数为：", len(train_vocab)) # 89487
print("前100个词 ： \n",list(train_vocab)[:100])
'''
 ['Betrusted', 'seedy', 'intervenes', 'WOBURN', 'Sinful', 'Isles', 'Sahibabad', 'energise', 'unsubtle', 'Excused', 'repealed', 
  'catching', 'conflicts', 'Councilor', 'forcedhis', 'Spamtown', 'crystal', 'reformist', 'MORRISONS', 'Marketwatch', 'Dominant', 
  'Haitian', 'Robby', 'Dismantles', 'WYSIWYM', 'Discipline', 'Juninho', 'replayed', 'Incas', 'Castleton', 'rates', 'Flops', 'foodseller', 
  'Tourky', 'Steel', 'Xeni', 'firstvictory', 'LL', 'VADUZ', 'eBooks', 'Kmarts', 'puzzling', 'PointBet', 'Manningham', 'resort', 'beginnings', 
  '127', '78th', '2.03', 'evangelist', 'Joining', 'livestock', 'Isotopes', 'Tilt', 'YMCA', 'Lodging', 'Virginity', 'Badrani', '1430', 'outsell', 
  'Arundel', 'Microsoystems', 'HAITI', 'gradually', 'SWIMMING', 'Tendulkar', 'CHELSEA', 'NRPB', 'ponders', 'Breakthroughs', 'stellar', 'Rheal', 
  'Boxing', '180m', 'Buccaneers', 'cantonment', 'Bosnian', 'defeated', 'wailing', 'results', 'Schelo', '20M', 'Bunting', 'reservoir', 'Buyers', 
  'SP5', 'Minority', 'Portraits', 'ESA', 'Provides', 'OFFERED', 'teddy', 'Kick', 'unconditional', 'ratty', 'eventual', 'RICK', 'Lamina', 'Bald', 'Sfor']
'''



'''
绘制词云
'''
# 使用jieba中的词性标注功能
import jieba.posseg as pseg

def get_a_list(text):
    """用于获取形容词列表"""
    # 使用jieba的词性标注方法切分文本,获得具有词性属性flag和词汇属性word的对象, 
    # 从而判断flag是否为形容词,来返回对应的词汇
    r = []
    for g in pseg.lcut(text):
        if g.flag == "a":
            r.append(g.word)
    return r

# 导入绘制词云的工具包
from wordcloud import WordCloud

def get_word_cloud(keywords_list, name):
    # 实例化绘制词云的类, 其中参数font_path是字体路径, 为了能够显示中文, 
    # max_words指词云图像最多显示多少个词, background_color为背景颜色 
    wordcloud = WordCloud(font_path="./fonts/SIMFANG.TTF", max_words=100, background_color="white")
    # 将传入的列表转化成词云生成器需要的字符串形式
    keywords_string = " ".join(keywords_list)
    # 生成词云
    print("keyword:",keywords_string)
    wordcloud.generate(keywords_string)

    # 绘制图像并显示
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    plt.savefig("./image/wordcloud_{}.png".format(name))

# 获得训练集上正样本
p_train_data = train_data[train_data["label"]==1]["sentence"]

# 对正样本的每个句子的词
train_p_a_vocab = chain(*map(lambda x: x, p_train_data))
#print(train_p_n_vocab)

# 调用绘制词云函数
get_word_cloud(train_p_a_vocab, "a")