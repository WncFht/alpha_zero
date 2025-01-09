import logging
import wandb
import math
import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

from game import *
logging.basicConfig(level = logging.INFO)
log = logging.getLogger(__name__)

# Initialize wandb
wandb.init(project="alphazero")

class MCTS():
    """
    这个类实现了蒙特卡洛树搜索(MCTS)算法。
    MCTS是一种用于决策的搜索算法,通过模拟来评估不同动作的价值。
    """

    def __init__(self, game, nnet, args):
        self.game = game  # 游戏环境
        self.nnet = nnet  # 神经网络模型
        self.args = args  # 配置参数
        self.Qsa = {}  # 存储状态-动作对(s,a)的Q值
        self.Nsa = {}  # 存储状态-动作对(s,a)被访问的次数
        self.Ns = {}   # 存储状态s被访问的次数
        self.Ps = {}   # 存储神经网络返回的初始策略(先验概率)

        self.Es = {}   # 存储状态s是否为终止状态
        self.Vs = {}   # 存储状态s的合法动作

    def getActionProb(self, canonicalBoard, temp=1):
        """
        执行多次MCTS模拟,并返回动作概率分布。

        参数:
            canonicalBoard: 标准化的棋盘状态
            temp: 温度参数,控制探索程度
                 temp=1时正常采样
                 temp→0时趋向于选择访问次数最多的动作
                 temp→∞时趋向于均匀随机选择

        返回:
            probs: 动作概率向量,其中第i个动作的概率正比于(访问次数)**(1/temp)
        """
        # 执行MCTS模拟
        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard)

        # 获取当前状态的字符串表示
        s = self.game.stringRepresentation(canonicalBoard)
        # 获取所有动作的访问次数
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        # 当温度为0时,选择访问次数最多的动作
        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        # 根据温度参数计算动作概率
        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, canonicalBoard):
        """
        执行一次MCTS(蒙特卡洛树搜索)迭代。该函数会递归调用直到找到叶子节点。
        在每个节点选择的动作都是具有最大上限置信界(UCB)值的动作,如论文所述。

        当找到叶子节点时,会调用神经网络返回初始策略P和状态值v。
        这个值会沿着搜索路径向上传播。如果叶子节点是终止状态,
        则将结果沿搜索路径向上传播。同时更新Ns、Nsa、Qsa的值。

        注意:由于v在[-1,1]范围内,如果v是当前玩家的状态值,
        那么对于另一个玩家来说其值为-v。

        参数:
            canonicalBoard: 当前游戏局面的标准形式

        返回:
            v: 当前标准局面的价值
        """

        # 获取当前局面的字符串表示
        s = self.game.stringRepresentation(canonicalBoard)

        # 检查是否为终止状态
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] is not None:
            # 如果是终止节点,直接返回游戏结果
            return self.Es[s]

        # 处理叶子节点
        if s not in self.Ps:
            # 使用神经网络预测策略和价值
            self.Ps[s], v = self.nnet.predict(canonicalBoard)
            # 获取有效移动
            valids = self.game.getValidMoves(canonicalBoard, 1)
            # 将无效移动的概率置为0
            self.Ps[s] = self.Ps[s] * valids  
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                # 重新归一化概率
                self.Ps[s] /= sum_Ps_s  
            else:
                # 如果所有有效移动都被屏蔽,则将所有有效移动设为等概率
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            # 存储有效移动和访问次数
            self.Vs[s] = valids
            self.Ns[s] = 0
            return v

        # 获取当前状态的有效移动
        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # 选择具有最大UCB值的动作
        for a in range(self.game.getActionSize()):
            if valids[a]:
                # 计算UCB值:Q值 + cpuct * 先验概率 * sqrt(父节点访问次数)/(1 + 当前动作访问次数)
                u = self.Qsa.get((s, a), 0) + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa.get((s, a), 0))

                if u > cur_best:
                    cur_best = u
                    best_act = a

        # 执行选中的动作
        a = best_act
        # 获取下一个状态和玩家
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        # 将下一个状态转换为标准形式
        next_s = self.game.getCanonicalForm(next_s, next_player)

        # 递归搜索下一个状态,注意要取负号因为是对手的视角
        v = -self.search(next_s)

        # 更新Q值和访问次数
        if (s, a) in self.Qsa:
            # 更新现有的Q值: Q = (N*Q + v)/(N+1)
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            # 新的状态-动作对
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        # 增加状态的访问次数
        self.Ns[s] += 1
        return v

class OthelloNNet(nn.Module):
    def __init__(self, game, args):
        # 游戏参数
        self.board_x, self.board_y = game.getBoardSize()  # 获取棋盘大小
        self.action_size = game.getActionSize()  # 获取动作空间大小
        self.args = args  # 存储参数

        super(OthelloNNet, self).__init__()
        # 定义卷积层
        self.conv1 = nn.Conv2d(1, args.num_channels, 3, stride=1, padding=1)  # 第一层卷积,输入通道1,输出通道num_channels,kernel_size=3
        self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)  # 第二层卷积
        self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)  # 第三层卷积
        self.conv4 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)  # 第四层卷积

        # 定义批归一化层
        self.bn1 = nn.BatchNorm2d(args.num_channels)  # 第一层批归一化
        self.bn2 = nn.BatchNorm2d(args.num_channels)  # 第二层批归一化
        self.bn3 = nn.BatchNorm2d(args.num_channels)  # 第三层批归一化
        self.bn4 = nn.BatchNorm2d(args.num_channels)  # 第四层批归一化

        # 定义全连接层
        self.fc1 = nn.Linear(args.num_channels*(self.board_x-4)*(self.board_y-4), 1024)  # 第一层全连接,输出1024维
        self.fc_bn1 = nn.BatchNorm1d(1024)  # 全连接层1的批归一化

        self.fc2 = nn.Linear(1024, 512)  # 第二层全连接,输出512维
        self.fc_bn2 = nn.BatchNorm1d(512)  # 全连接层2的批归一化

        self.fc3 = nn.Linear(512, self.action_size)  # 策略头,输出动作概率
        self.fc4 = nn.Linear(512, 1)  # 价值头,输出状态价值

    def forward(self, s):
        # 前向传播函数
        s = s.view(-1, 1, self.board_x, self.board_y)                # 将输入重塑为 batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))                          # 第一层卷积+批归一化+ReLU
        s = F.relu(self.bn2(self.conv2(s)))                          # 第二层卷积+批归一化+ReLU
        s = F.relu(self.bn3(self.conv3(s)))                          # 第三层卷积+批归一化+ReLU
        s = F.relu(self.bn4(self.conv4(s)))                          # 第四层卷积+批归一化+ReLU
        s = s.view(-1, self.args.num_channels*(self.board_x-4)*(self.board_y-4))  # 展平特征图

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)  # 全连接层1+批归一化+ReLU+Dropout
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)  # 全连接层2+批归一化+ReLU+Dropout

        pi = self.fc3(s)                                                                         # 计算策略(动作概率)
        v = self.fc4(s)                                                                          # 计算价值

        return F.log_softmax(pi, dim=1), torch.tanh(v)  # 返回log概率和tanh后的价值


class AverageMeter(object):
    """用于计算和存储平均值和当前值的类,来自PyTorch官方示例"""

    def __init__(self):
        self.val = 0  # 当前值
        self.avg = 0  # 平均值 
        self.sum = 0  # 总和
        self.count = 0  # 计数器

    def __repr__(self):
        return f'{self.avg:.2e}'  # 返回平均值的科学计数法表示

    def update(self, val, n=1):
        """更新统计值
        Args:
            val: 当前值
            n: 样本数量,默认为1
        """
        self.val = val  # 更新当前值
        self.sum += val * n  # 更新总和
        self.count += n  # 更新计数
        self.avg = self.sum / self.count  # 计算新的平均值


class NNetWrapper():
    """神经网络包装类,用于训练和预测"""
    
    def __init__(self, game, args):
        """初始化神经网络
        Args:
            game: 游戏实例
            args: 配置参数
        """
        self.nnet = OthelloNNet(game, args)  # 创建神经网络实例
        self.board_x, self.board_y = game.getBoardSize()  # 获取棋盘尺寸
        self.action_size = game.getActionSize()  # 获取动作空间大小
        self.args = args  # 保存配置参数

        if args.cuda:  # 如果使用GPU
            self.nnet.cuda()  # 将网络移至GPU

    def train(self, examples):
        """训练神经网络
        Args:
            examples: 训练样本列表,每个样本包含(board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters(), lr=self.args.lr)  # 创建Adam优化器

        wandb.config.update(self.args)  # 更新wandb配置

        for epoch in range(self.args.epochs):  # 训练指定轮数
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()  # 设置为训练模式
            pi_losses = AverageMeter()  # 策略损失统计
            v_losses = AverageMeter()  # 价值损失统计

            batch_count = int(len(examples) / self.args.batch_size)  # 计算批次数

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                # 随机采样一个批次
                sample_ids = np.random.randint(len(examples), size=self.args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                # 转换为张量
                boards = torch.FloatTensor(np.array(boards).astype(np.float32))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float32))

                if self.args.cuda:  # 如果使用GPU
                    boards, target_pis, target_vs = boards.cuda(), target_pis.cuda(), target_vs.cuda()

                # 前向传播
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)  # 计算策略损失
                l_v = self.loss_v(target_vs, out_v)  # 计算价值损失
                total_loss = l_pi + l_v  # 总损失

                # 记录损失
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                # 记录到wandb
                wandb.log({"pi_loss": pi_losses.avg, "v_loss": v_losses.avg})

                # 反向传播和优化
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, board):
        """预测给定棋盘状态的策略和价值
        Args:
            board: 棋盘状态的numpy数组
        Returns:
            策略概率分布和状态价值
        """
        board = torch.FloatTensor(board.astype(np.float32))  # 转换为张量
        if self.args.cuda: board = board.cuda()  # 移至GPU
        board = board.view(1, self.board_x, self.board_y)  # 调整维度
        self.nnet.eval()  # 设置为评估模式
        with torch.no_grad():
            pi, v = self.nnet(board)  # 进行预测

        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        """计算策略损失函数
        Args:
            targets: 目标策略
            outputs: 预测策略
        """
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        """计算价值损失函数
        Args:
            targets: 目标价值
            outputs: 预测价值
        """
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        """保存模型检查点
        Args:
            folder: 保存文件夹
            filename: 保存文件名
        """
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        """加载模型检查点
        Args:
            folder: 加载文件夹
            filename: 加载文件名
        """
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ValueError("No model in path {}".format(filepath))
        map_location = None if self.args.cuda else 'cpu'  # 根据是否使用GPU决定加载位置
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])  # 加载模型参数


class SelfPlay():
    """自我对弈类,用于执行自我对弈和学习"""

    def __init__(self, game, nnet, args):
        """初始化自我对弈类
        Args:
            game: 游戏实例
            nnet: 神经网络实例
            args: 配置参数
        """
        self.game = game  # 游戏实例
        self.nnet = nnet  # 当前网络
        self.pnet = self.nnet.__class__(self.game, args)  # 对手网络
        self.args = args  # 配置参数
        self.mcts = MCTS(self.game, self.nnet, self.args)  # 蒙特卡洛树搜索实例
        self.trainExamplesHistory = []  # 训练样本历史,保存最近几次迭代的样本

    def executeEpisode(self):
        """执行一局自我对弈
        从玩家1开始,每一步都作为训练样本添加到trainExamples中。
        游戏结束后,根据游戏结果为每个样本分配价值。
        当episodeStep < tempThreshold时使用temp=1,之后使用temp=0。

        Returns:
            trainExamples: 训练样本列表,每个样本包含(canonicalBoard, pi, v)
        """
        trainExamples = []  # 训练样本列表
        board = self.game.getInitBoard()  # 获取初始棋盘
        self.curPlayer = 1  # 当前玩家
        episodeStep = 0  # 当前步数

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)  # 获取标准形式的棋盘
            temp = int(episodeStep < self.args.tempThreshold)  # 根据步数决定温度参数

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)  # 获取动作概率
            sym = self.game.getSymmetries(canonicalBoard, pi)  # 获取对称的状态和策略
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])  # 添加训练样本

            action = np.random.choice(len(pi), p=pi)  # 根据概率选择动作
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)  # 执行动作

            r = self.game.getGameEnded(board, self.curPlayer)  # 检查游戏是否结束

            if r is not None:
                # 根据游戏结果分配价值:胜者为1,败者为-1,平局为0
                return [(x[0], x[2], r * (1 if self.curPlayer == x[1] else -1)) for x in trainExamples]

    def learn(self):
        """执行学习过程
        执行numIters次迭代,每次迭代包含numEps局自我对弈。
        每次迭代后,用trainExamples中的样本重新训练神经网络。
        然后将新网络与旧网络对弈,只有在胜率超过updateThreshold时才接受新网络。
        """

        for i in range(1, self.args.numIters + 1):
            log.info(f'Starting Iter #{i} ...')  # 记录开始新的迭代
            iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)  # 当前迭代的训练样本

            for _ in tqdm(range(self.args.numEps), desc="Self Play"):  # 执行自我对弈
                self.mcts = MCTS(self.game, self.nnet, self.args)  # 重置搜索树
                iterationTrainExamples += self.executeEpisode()  # 执行一局对弈并收集样本

            self.trainExamplesHistory.append(iterationTrainExamples)  # 保存当前迭代的样本

            # 如果历史样本数量超过限制,删除最早的样本
            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)

            # 打乱所有训练样本
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # 保存当前网络,训练新网络
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args)  # 创建对手的MCTS

            self.nnet.train(trainExamples)  # 训练新网络
            nmcts = MCTS(self.game, self.nnet, self.args)  # 创建新网络的MCTS

            log.info('PITTING AGAINST PREVIOUS VERSION')  # 开始新旧网络对弈
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)  # 进行对弈

            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))  # 记录对弈结果
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                log.info('REJECTING NEW MODEL')  # 如果新网络表现不够好,拒绝更新
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')  # 接受新网络
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')


class dotdict(dict):
    """继承dict类,允许使用点号访问字典元素"""
    def __getattr__(self, name):
        return self[name]


args = dotdict({
    # 神经网络训练相关参数
    'lr': 0.001,               # 学习率
    'dropout': 0.1,            # dropout比率
    'epochs': 10,              # 训练轮数
    'batch_size': 64,          # 批大小
    'cuda': torch.cuda.is_available(),  # 是否使用GPU
    'num_channels': 512,       # 卷积层通道数

    # AlphaZero算法相关参数
    'numIters': 200,           # 总迭代次数
    'numEps': 100,             # 每次迭代中自我对弈的局数
    'tempThreshold': 15,       # 温度阈值,用于控制探索
    'updateThreshold': 0.6,    # 新网络被接受的胜率阈值
    'maxlenOfQueue': 200000,   # 训练样本队列的最大长度
    'numItersForTrainExamplesHistory': 20,  # 保留的历史训练样本的迭代次数
    'numMCTSSims': 25,         # 每步MCTS模拟的次数
    'arenaCompare': 40,        # 新旧网络对弈的局数
    'cpuct': 1,                # MCTS探索常数

    # 模型保存相关参数
    'checkpoint': './temp/',   # 检查点保存路径
    'load_model': False,       # 是否加载已有模型
    'load_folder_file': ('./temp/','best.pth.tar'),  # 加载模型的路径和文件名
    })

def main():
    """主函数,处理命令行参数并执行训练或对弈"""
    import argparse
    parser = argparse.ArgumentParser()
    # 训练相关参数
    parser.add_argument('--train', action="store_true")  # 是否训练模型
    parser.add_argument('--board_size', type=int, default=6)  # 棋盘大小
    
    # 对弈相关参数
    parser.add_argument('--play', action="store_true")  # 是否进行对弈
    parser.add_argument('--verbose', action="store_true")  # 是否显示详细信息
    parser.add_argument('--round', type=int, default=2)  # 对弈轮数
    parser.add_argument('--player1', type=str, default='human', choices=['human', 'random', 'greedy', 'alphazero'])  # 玩家1类型
    parser.add_argument('--player2', type=str, default='alphazero', choices=['human', 'random', 'greedy', 'alphazero'])  # 玩家2类型
    parser.add_argument('--ckpt_file', type=str, default='best.pth.tar')  # 加载的模型文件名
    
    # 解析命令行参数并更新args字典
    args_input = vars(parser.parse_args())
    for k,v in args_input.items():
        args[k] = v
    
    # 创建游戏实例
    g = OthelloGame(args.board_size)

    # 训练模式
    if args.train:
        nnet = NNetWrapper(g, args)  # 创建神经网络
        if args.load_model:  # 如果需要加载已有模型
            log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
            nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

        log.info('Loading the SelfCoach...')
        s = SelfPlay(g, nnet, args)  # 创建自我对弈实例

        log.info('Starting the learning process 🎉')
        s.learn()  # 开始训练
    
    # 对弈模式
    if args.play:
        def getPlayFunc(name):
            """根据玩家类型返回相应的对弈函数"""
            if name == 'human':
                return HumanOthelloPlayer(g).play  # 人类玩家
            elif name == 'random':
                return RandomPlayer(g).play  # 随机玩家
            elif name == 'greedy':
                return GreedyOthelloPlayer(g).play  # 贪心玩家
            elif name == 'alphazero':
                # AlphaZero玩家
                nnet = NNetWrapper(g, args)
                nnet.load_checkpoint(args.checkpoint, args.ckpt_file)
                mcts = MCTS(g, nnet, dotdict({'numMCTSSims': 50, 'cpuct':1.0}))
                return lambda x: np.argmax(mcts.getActionProb(x, temp=0))
            else:
                raise ValueError('not support player name {}'.format(name))
                
        # 获取两个玩家的对弈函数
        player1 = getPlayFunc(args.player1)
        player2 = getPlayFunc(args.player2)
        # 创建对弈场景并开始对弈
        arena = Arena(player1, player2, g, display=OthelloGame.display)
        results = arena.playGames(args.round, verbose=args.verbose)
        print("Final results: Player1 wins {}, Player2 wins {}, Draws {}".format(*results))

if __name__ == '__main__':
    main()
