import torch
import torchtext
# 导入torchtext.datasets中的文本分类任务
from torchtext.datasets import text_classification
import os
from model import TextSentiment
from trainer import Trainer
import argparse
from torch.utils.data import DataLoader

#文字识别时，文本行图像长度不一，需要自定义整理。将一个batch的数据拼成一个向量
def generate_batch(batch):
    """
    description: 生成batch数据函数
    :param batch: 由样本张量和对应标签的元组组成的batch_size大小的列表
                形如:
                [(label1, sample1), (lable2, sample2), ..., (labelN, sampleN)]
    return: 样本张量和标签各自的列表形式(张量)
            形如:
            text = tensor([sample1, sample2, ..., sampleN])
            label = tensor([label1, label2, ..., labelN])
    """
    # 从batch中获得标签张量
    label = torch.tensor([entry[0] for entry in batch])
    # 从batch中获得样本张量
    text = [entry[1] for entry in batch]
    text = torch.cat(text)
    # 返回结果
    return text, label

# if __name__ == "__main__":
parser = argparse.ArgumentParser(description = 'News classification')
parser.add_argument('--load_data_path', type = str, default = "./data/", metavar = 'data path')   
parser.add_argument('--checkpoint_dir', type = str, default = "./model/", metavar = 'checkpoint path')   
parser.add_argument('--batch_size', type = int, default = 16, metavar = 'batchsize')
parser.add_argument('--embedded_dim', type = int, default = 32, metavar = 'embedded dimension / 词嵌入维度') 
args = parser.parse_args()

load_data_path = args.load_data_path
batch_size = args.batch_size
embedded_dim = args.embedded_dim
checkpoint_dir = args.checkpoint_dir

'''
数据导入
'''
# 如果不存在该路径, 则创建这个路径
if not os.path.isdir(load_data_path):
    os.mkdir(load_data_path)

# 选取torchtext中的文本分类数据集'AG_NEWS'即新闻主题分类数据, print保存在指定目录下
# 并将数值映射后的训练和验证数据加载到内存中
train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](root=load_data_path) 
'''
len(train_dataset)=120000, len(test_dataset)=7600
train_dataset._data[0] : (2, tensor([  432,   426,     2,  1606, 14839,   114,    67,     3,   849,    14,
           28,    15,    28,    16, 50726,     4,   432,   375,    17,    10,
        67508,     7, 52259,     4,    43,  4010,   784,   326,     2]))
len(train_dataset[0][1]):29
train_dataset._labels : {0, 1, 2, 3}
train_dataset 最大值：95811，
text_classification.DATASETS根据数据构建词表，采用n_gram，一共95812个词
训练集的不同词个数是89487，多出来的是3_gram的结果
'''


'''
加载训练好的模型
'''
# 进行可用设备检测, 有GPU的话将优先使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)
# 获得整个语料包含的不同词汇总数
VOCAB_SIZE = len(train_dataset.get_vocab())
# 获得类别总数
NUN_CLASS = len(train_dataset.get_labels())
# 实例化模型
model = TextSentiment(VOCAB_SIZE, embedded_dim, NUN_CLASS)
#加载模型
model.load_state_dict(torch.load("{}/model.pth".format(checkpoint_dir)))
model.cuda()

'''
测试
'''
# 初始化验证损失和准确率为0
loss = 0
acc = 0
criterion = torch.nn.CrossEntropyLoss().to(device)
# 和训练相同, 使用DataLoader获得训练数据生成器
test_data = DataLoader(test_dataset, batch_size=batch_size, collate_fn=generate_batch)
# 按批次取出数据验证
i=0
for text, cls in test_data:
    print("{} , {}".format(i, text.shape))
    text = text.cuda()#torch.Size([691])  691是一个bacth(16样本)拼在一起的长度
    '''
    此处还没有进行word2vec或者embedding
    tensor([ 2148,  6081,     4,  3010,  5246, 15496, 14671,   586,   783,     7,
           33,  1232,   347,     8,     3,   128,   512,     5,   171,     3,
         2145,  2148,     5,    31, 34943,   147,    39,     3,   738,  3010,
            4,    76,  2684,   131,   616,  4930,  2405,     5,    24,  2556,
          759,    66,     2, 10783,   349,  3313,   940,    44,   666,    24,
        62208,     3,  3359,   140,   349,   242,    22, 11057,  6706,     8,
           23,  2692,   134,     7,  2123,    24,  6742,     4,   224,  6703,
         1557,  3557,  1031,     2,   837,    90,  2572,    16,  2163,    21,
          836,   836,     4,  7774,    14,   263,   201,    15,    16,  5032,
          372,    13,  3244,    87,   452,   347,     4,   205,  2242,     4,
          473,  4472,     9,    95,  4336,     9, 10932,  5879,   662,   131,
        20398,    19,   694,   347,     4,    20,     3,   836,  1556,  9190,
            3,  2163,  4096,     4, 27430,     2,     0, 18684,     8,  1321,
          132,   205,    32,    16,   425,   454,    76,   748,  5958,     6,
         1698,     7,  2151,  3054, 18230, 26150,    40,    82,  4116,    12,
          565,     3,   128,   144,  1446,  2450,  8576,     7,   141,     2,
          179,  2571,  1215,   399,     4,   747,  2398,   119,   179,  2571,
           43, 15102,     5,   333,    13,   917,   584,     8,  2123,   399,
         1514,    11,     3,   225,     4,   224, 13824,    18,   230,    43,
         5829,  2398,    18,   799,     5,    38,  4120,     2,     3,  3743,
          911,  4216,     7, 26545,   178,   515,   377,    60,    24,    74,
           16,     3,   606,  4879,    80,    11,     4,     3,  2778,  3275,
            9,     3, 51446,     7,  2778, 21087,     8,    33,   446, 35634,
         2184,  2125,    16,    46,    63,    12,   443,     2,  7736,  3171,
           11,   663, 12205,    16,     3,  1454,   916,  3913,  7736,     4,
            9,     3,  3705,   228,    94,    38,    11,    33,   250,     2,
         1437,  3705,  3755,  1164,  6842, 43091,    27,    66,    18,    45,
           31,   374,    22,  1168,  4421,     4,    45,     3,  1454,    43,
         3399,     5, 27681, 62480,   200,   126,  2760,     8,  1443,    64,
           52,     2,    10,     2,  1280,    14,   365,   329,    15,   365,
          329,    16, 30977, 32741,     4, 31125,    14,    32,    15,    16,
            3,   200,   126,     7,     3,  9561,   512,     7,  6094,  2368,
           66,     4,     6,   133,    35,     3,    52,     2,    10,     2,
          101,     9,  6094,    17,    10,   132,   139,  5408, 15646,  6209,
        12670,    12,  2615,     5,  1685,     9,  1187,    39,   270,  2312,
         2006,     5,     3,   287,  4247,     2,  1739, 40294,   122,  3694,
            3,   169,   279,    17,    10,   525,     5,   663,  3555,   174,
           19,  1245,  5271,     6,  3581,  3315,     2, 26502,  1864,     5,
          171,  7502,  1945,   689, 20264, 38089, 26502,  1145,    49,  1140,
        50423,     4,   335,  5777,  4734,    19,     3,   133,    13,    10,
           48,  1501,     4,     5,   171,     6,  1945,  9636,     8,     3,
          128,     9,   217,   290,    64,   197,    11,   115,     2,  1292,
         3068, 10503,    12,   138,     2,   421,   145,    54,  5815,   750,
            7,   130,  1780,  1868,     2,  1292,    27,    75,    50,  1247,
           17,    84,  8832,   278,     3,   385,   446,    12,     3,   129,
          791,    17,    10,   139,   684,     5,   270,     8,    71,     2,
           50,   457,  2208, 10503,    22,   565, 18931,     8,  5568,   280,
         1741,     8,   340,     9,   267,   421,     4,    50,    27,     8,
           31,  1797,     2,   436,     3,    51,  4498,   400,  1431,  4861,
         5042,    36,  8011,    19,     3,   144,  9344,  1047, 12825,     4,
          104,    22,  8269,     8,  2078,  3109,     4,     6,  2808,    76,
        19616,    33,  2209,    27,    66,     2,  9146,    22,  4873,    19,
         9779,    19,     3,   824,     7,  8370,  3460,  1805, 10694,  1577,
           54,    68,   603,  2184,  2125,     5,     3,  1493,     4,  2001,
           11,  1657,  7045,     4,     9,    43,  2368,     5,  4029,  1046,
         1873,     6,  3976,    54,     3,   196,   301,  1576,   326,    70,
          116,     4,   861,     5,  1624,    19,   379,  2078,  2900,  9779,
           11,     6,  3599,   122,    18,    34,   355,     3, 11030,     8,
            3,  6293,     7,  4864,     2,     2,     2,  2125,  1790,  2945,
          318,  1710,  2958,   678,   685,  3771,   631,    98,   759,     4,
         2184,  2125,    29,   292,     3,   196,   301,  2045,  2945,    69,
            3,   250,    50,    36,  5196,    25,   544,     2,   249,     7,
         2699,  2440,     8,  1037,   190,  9764,   117,  5381,    42,   165,
            2,     2,     2,    42,   164,  1037,  1788,     8,     3,    89,
          160,    40,   683,    18, 20270,  9764,     9,  5381,   105,    38,
         1981,     5,   198,  9920,   898,     9,   394,    63,    38, 18292,
            2,  1526,   536,    67, 11029,  1526,   155,     5,   275,     3,
         6157,     9,   161,  4009,   158,  7231,     2,   173,    12, 14747,
           83,     4,  2070,     5,    23,   792, 28450, 43337,     7,  3065,
            2], device='cuda:0')
    '''
    cls = cls.cuda()
    # 验证阶段, 不再求解梯度
    with torch.no_grad():
        # 使用模型获得输出
        output = model(text)#([16,4])
        '''
        tensor([[-1.0066,  3.3920, -1.3146, -1.4207],
        [-1.6859,  0.6940,  1.4173, -0.4782],
        [-0.3828,  2.8236, -1.2782, -1.7264],
        [-0.1134, -1.5396, -2.0525,  4.1792],
        [-0.5464, -0.8471, -0.2831,  1.9133],
        [ 0.2830,  1.3561, -1.1232, -0.8889],
        [ 1.4966, -1.1322,  0.0763, -0.6416],
        [ 1.7531, -0.9504, -0.6131, -0.7426],
        [ 2.1410, -1.5754, -0.1645, -0.8592],
        [ 0.7200,  1.0223, -1.3232, -0.8003],
        [ 2.3062, -1.4224, -0.7440, -0.5916],
        [ 1.3371, -0.9268, -0.6980,  0.0277],
        [-1.3993,  2.5020, -1.1515, -0.1203],
        [-0.4408,  1.5124, -0.6586, -0.7175],
        [-0.1292, -0.2678,  0.6120, -0.1454],
        [-0.7599, -1.0789,  0.5970,  1.3503]], device='cuda:0')
        '''
        # 计算损失
        loss = criterion(output, cls)
        # 将损失和准确率加到总损失和准确率中
        loss += loss.item()
        acc += (output.argmax(1) == cls).sum().item()
        i+=1

# 返回本轮验证的平均损失和平均准确率
test_loss, test_acc = loss / len(test_dataset), acc / len(test_dataset)
print("Test : loss {} | acc {}".format(test_loss, test_acc))


'''
打印Embedding层参数
'''
# 打印从模型的状态字典中获得的Embedding矩阵
print(model.state_dict()['embedding.weight'].shape)#([95812, 32])
print(model.state_dict()['embedding.weight'])