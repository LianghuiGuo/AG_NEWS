
import torch
import time
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, train_data, val_data, model, batch_size, epoch, device):
        self.train_data = train_data
        self.val_data = val_data
        self.model = model
        self.batch_size = batch_size
        self.epoch = epoch
        self.device = device
        # 选择损失函数, 这里选择预定义的交叉熵损失函数
        self.criterion = torch.nn.CrossEntropyLoss().to(device)
        # 选择随机梯度下降优化器
        self.optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
        # 选择优化器步长调节方法StepLR, 用来衰减学习率
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.9)
        
    def run_train(self):
        """开始训练"""
        for epoch in range(self.epoch):
            # 记录概论训练的开始时间
            start_time = time.time()
            # 调用train和valid函数得到训练和验证的平均损失, 平均准确率
            train_loss, train_acc = self.train(self.train_data)
            valid_loss, valid_acc = self.valid(self.val_data)

            # 计算训练和验证的总耗时(秒)
            secs = int(time.time() - start_time)
            # 用分钟和秒表示
            mins = secs / 60
            secs = secs % 60

            # 打印训练和验证耗时，平均损失，平均准确率
            print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
            print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
            print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')
            
    def train(self, train_data):
        """模型训练函数"""
        # 初始化训练损失和准确率为0
        train_loss = 0
        train_acc = 0

        # 使用数据加载器生成BATCH_SIZE大小的数据进行批次训练
        # data就是N多个generate_batch函数处理后的BATCH_SIZE大小的数据生成器
        data = DataLoader(train_data, batch_size=self.batch_size, shuffle=True,
                        collate_fn=self.generate_batch)

        # 对data进行循环遍历, 使用每个batch的数据进行参数更新
        for i, (text, cls) in enumerate(data):
            text = text.cuda()
            cls = cls.cuda()
            # 设置优化器初始梯度为0
            self.optimizer.zero_grad()
            # 模型输入一个批次数据, 获得输出
            output = self.model(text)
            # 根据真实标签与模型输出计算损失
            loss = self.criterion(output, cls)
            # 将该批次的损失加到总损失中
            train_loss += loss.item()
            # 误差反向传播
            loss.backward()
            # 参数进行更新
            self.optimizer.step()
            # 将该批次的准确率加到总准确率中
            train_acc += (output.argmax(1) == cls).sum().item()

        # 调整优化器学习率  
        self.scheduler.step()

        # 返回本轮训练的平均损失和平均准确率
        return train_loss / len(train_data), train_acc / len(train_data)

    def valid(self, valid_data):
        """模型验证函数"""
        # 初始化验证损失和准确率为0
        loss = 0
        acc = 0

        # 和训练相同, 使用DataLoader获得训练数据生成器
        data = DataLoader(valid_data, batch_size=self.batch_size, collate_fn=self.generate_batch)
        # 按批次取出数据验证
        for text, cls in data:
            text = text.cuda()
            cls = cls.cuda()
            # 验证阶段, 不再求解梯度
            with torch.no_grad():
                # 使用模型获得输出
                output = self.model(text)
                # 计算损失
                loss = self.criterion(output, cls)
                # 将损失和准确率加到总损失和准确率中
                loss += loss.item()
                acc += (output.argmax(1) == cls).sum().item()

        # 返回本轮验证的平均损失和平均准确率
        return loss / len(valid_data), acc / len(valid_data)

    def generate_batch(self, batch):
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