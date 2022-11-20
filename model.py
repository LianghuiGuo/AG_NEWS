# 导入必备的torch模型构建工具
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# 指定BATCH_SIZE的大小
BATCH_SIZE = 16

# 进行可用设备检测, 有GPU的话将优先使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextSentiment(nn.Module):
    """文本分类模型"""
    def __init__(self, vocab_size, embed_dim, num_class):
        """
        description: 类的初始化函数
        :param vocab_size: 整个语料包含的不同词汇总数
        :param embed_dim: 指定词嵌入的维度
        :param num_class: 文本分类的类别总数
        """ 
        super().__init__()
        # 实例化embedding层, sparse=True代表每次对该层求解梯度时, 只更新部分权重.
        self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=True)
        # 实例化线性层, 参数分别是embed_dim和num_class.
        self.fc = nn.Linear(embed_dim, num_class)
        # 为各层初始化权重
        self.init_weights()

    def init_weights(self):
        """初始化权重函数"""
        # 指定初始权重的取值范围数
        initrange = 0.5
        # 各层的权重参数都是初始化为均匀分布
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        # 偏置初始化为0
        self.fc.bias.data.zero_()

    def forward(self, text):
        """
        :param text: 文本数值映射后的结果 (m)
        :return: 与类别数尺寸相同的张量, 用以判断文本类别
        """
        # 获得embedding的结果embedded
        embedded = self.embedding(text) #(m, 32) 其中m是BATCH_SIZE大小的数据中词汇总数  ([691, 32])
        # 接下来我们需要将(m, 32)转化成(BATCH_SIZE, 32)
        # 以便通过fc层后能计算相应的损失
        # 首先, 我们已知m的值远大于BATCH_SIZE=16,
        # 用m整除BATCH_SIZE, 获得m中共包含c个BATCH_SIZE
        c = embedded.size(0) // BATCH_SIZE   #43  一个batch的平均句子长度
        # 之后再从embedded中取c*BATCH_SIZE个向量得到新的embedded
        # 这个新的embedded中的向量个数可以整除BATCH_SIZE
        embedded = embedded[:BATCH_SIZE*c] #(c*batch_size, 32)   torch.Size([688, 32])
        # 因为我们想利用平均池化的方法求embedded中指定行数的列的平均数,
        # 但平均池化方法是作用在行上的, 并且需要3维输入
        # 因此我们对新的embedded进行转置并拓展维度
        embedded = embedded.transpose(1, 0).unsqueeze(0) #(1,32,c*batch_size)   #torch.Size([1, 32, 688])
        # 然后就是调用平均池化的方法, 并且核的大小为c
        # 即取每c的元素计算一次均值作为结果
        embedded = F.avg_pool1d(embedded, kernel_size=c)   #torch.Size([1, 32, 16])
        print(embedded.shape)
        # 最后，还需要减去新增的维度, 然后转置回去输送给fc层
        return self.fc(embedded[0].transpose(1, 0))  #embedded[0].transpose(1, 0): (16,32)


# 获得整个语料包含的不同词汇总数
VOCAB_SIZE = 1000
# 指定词嵌入维度
EMBED_DIM = 32
# 获得类别总数
NUN_CLASS = 4
model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUN_CLASS).to(device)

data = Variable(torch.ones([691])).long().cuda() 
model(data)