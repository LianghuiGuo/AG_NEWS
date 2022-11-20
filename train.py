# 导入相关的torch工具包
import torch
import torchtext
# 导入torchtext.datasets中的文本分类任务
from torchtext.datasets import text_classification
import os
from model import TextSentiment
from trainer import Trainer
from torch.utils.data.dataset import random_split
import argparse

def main(args):
    load_data_path = args.load_data_path
    batch_size = args.batch_size
    embedded_dim = args.embedded_dim
    n_epoch = args.n_epoch
    checkpoint_dir = args.checkpoint_dir
    
    '''
    数据导入
    '''
    # 如果不存在该路径, 则创建这个路径
    if not os.path.isdir(load_data_path):
        os.mkdir(load_data_path)

    # 选取torchtext中的文本分类数据集'AG_NEWS'即新闻主题分类数据, 保存在指定目录下
    # 并将数值映射后的训练和验证数据加载到内存中
    train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](root=load_data_path)

    '''
    模型
    '''
    # 进行可用设备检测, 有GPU的话将优先使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(0)
    # 获得整个语料包含的不同词汇总数
    VOCAB_SIZE = len(train_dataset.get_vocab())#95812
    # 获得类别总数
    NUN_CLASS = len(train_dataset.get_labels())#4
    # 实例化模型
    model = TextSentiment(VOCAB_SIZE, embedded_dim, NUN_CLASS)
    model.cuda()
    
    '''
    训练
    '''
    # 定义初始的验证损失
    min_valid_loss = float('inf')
    # 从train_dataset取出0.95作为训练集, 先取其长度
    train_len = int(len(train_dataset) * 0.95)
    # 然后使用random_split进行乱序划分, 得到对应的训练集和验证集
    sub_train_, sub_valid_ = random_split(train_dataset, [train_len, len(train_dataset) - train_len])
    #开始训练
    trainer = Trainer(sub_train_, sub_valid_, model, batch_size, n_epoch, device)
    trainer.run_train()
    
    '''
    测试
    '''
    test_loss, test_acc = trainer.valid(test_dataset)
    print("Test : loss {} | acc {}".format(test_loss, test_acc))
    
    '''
    保存模型
    '''
    #mkdir and save weights
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    torch.save(model.state_dict(), "{}/model.pth".format(checkpoint_dir))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'News classification')
    parser.add_argument('--load_data_path', type = str, default = "./data/", metavar = 'data path')   
    parser.add_argument('--checkpoint_dir', type = str, default = "./model/", metavar = 'checkpoint path')   
    parser.add_argument('--batch_size', type = int, default = 16, metavar = 'batchsize')
    parser.add_argument('--embedded_dim', type = int, default = 32, metavar = 'embedded dimension / 词嵌入维度') 
    parser.add_argument('--n_epoch', type = int, default = 1, metavar = 'epochs') 
    args = parser.parse_args()
    main(args)