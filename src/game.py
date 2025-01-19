import numpy as np
import logging
from tqdm import tqdm
log = logging.getLogger(__name__)

class Board():
    '''
    作者: Eric P. Nichols
    日期: 2008年2月8日
    棋盘类。
    棋盘数据:
    1=白棋, -1=黑棋, 0=空格
    '''

    # 棋盘上8个方向的偏移量列表,以(x,y)表示
    __directions = [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]

    def __init__(self, n):
        """初始化棋盘配置"""

        self.n = n
        # 创建空棋盘数组
        self.pieces = [None]*self.n
        for i in range(self.n):
            self.pieces[i] = [0]*self.n

        # 放置初始的4个棋子
        self.pieces[int(self.n/2)-1][int(self.n/2)] = 1
        self.pieces[int(self.n/2)][int(self.n/2)-1] = 1
        self.pieces[int(self.n/2)-1][int(self.n/2)-1] = -1
        self.pieces[int(self.n/2)][int(self.n/2)] = -1

    # 添加[][]索引语法到Board类
    def __getitem__(self, index): 
        return self.pieces[index]

    def countDiff(self, color):
        """计算给定颜色的棋子数量差值
        (1表示白棋, -1表示黑棋, 0表示空格)"""
        count = 0
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y]==color:
                    count += 1
                if self[x][y]==-color:
                    count -= 1
        return count

    def get_legal_moves(self, color):
        """返回给定颜色的所有合法移动
        (1表示白棋, -1表示黑棋)
        """
        moves = set()  # 存储合法移动

        # 获取所有给定颜色的棋子位置
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y]==color:
                    newmoves = self.get_moves_for_square((x,y))
                    moves.update(newmoves)
        return list(moves)

    def has_legal_moves(self, color):
        """检查给定颜色是否有合法移动"""
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y]==color:
                    newmoves = self.get_moves_for_square((x,y))
                    if len(newmoves)>0:
                        return True
        return False

    def get_moves_for_square(self, square):
        """返回以给定方格为基础的所有合法移动。
        例如,如果给定方格是(3,4)且包含黑棋,
        而(3,5)和(3,6)包含白棋,且(3,7)为空,
        则返回的移动之一是(3,7),因为从那里到(3,4)的所有棋子都会被翻转。
        """
        (x,y) = square

        # 确定棋子颜色
        color = self[x][y]

        # 跳过空源方格
        if color==0:
            return None

        # 搜索所有可能的方向
        moves = []
        for direction in self.__directions:
            move = self._discover_move(square, direction)
            if move:
                moves.append(move)

        # 返回生成的移动列表
        return moves

    def execute_move(self, move, color):
        """在棋盘上执行给定的移动;根据需要翻转棋子。
        color给出要下的棋子颜色(1=白棋,-1=黑棋)
        """

        # 类似于移动生成,从新棋子的方格开始,
        # 沿8个方向寻找允许翻转的棋子。

        flips = [flip for direction in self.__directions
                      for flip in self._get_flips(move, direction, color)]
        assert len(list(flips))>0
        for x, y in flips:
            self[x][y] = color

    def _discover_move(self, origin, direction):
        """返回合法移动的终点,从给定起点开始,按给定增量移动。"""
        x, y = origin
        color = self[x][y]
        flips = []

        for x, y in Board._increment_move(origin, direction, self.n):
            if self[x][y] == 0:
                if flips:
                    return (x, y)
                else:
                    return None
            elif self[x][y] == color:
                return None
            elif self[x][y] == -color:
                flips.append((x, y))

    def _get_flips(self, origin, direction, color):
        """获取用于execute_move函数的顶点和方向的翻转列表"""
        # 初始化变量
        flips = [origin]

        for x, y in Board._increment_move(origin, direction, self.n):
            if self[x][y] == 0:
                return []
            if self[x][y] == -color:
                flips.append((x, y))
            elif self[x][y] == color and len(flips) > 0:
                return flips

        return []

    @staticmethod
    def _increment_move(move, direction, n):
        """移动增量的生成器表达式"""
        move = list(map(sum, zip(move, direction)))
        while all(map(lambda x: 0 <= x < n, move)): 
            yield move
            move=list(map(sum,zip(move,direction)))

class OthelloGame():
    """黑白棋游戏类,实现游戏规则和逻辑"""
    
    # 棋子的显示字符映射
    square_content = {
        -1: "X",  # 黑棋
        +0: "-",  # 空格
        +1: "O"   # 白棋
    }

    @staticmethod
    def getSquarePiece(piece):
        """获取棋子的显示字符
        Args:
            piece: 棋子值(-1,0,1)
        Returns:
            对应的显示字符
        """
        return OthelloGame.square_content[piece]

    def __init__(self, n):
        """初始化游戏
        Args:
            n: 棋盘大小(n x n)
        """
        self.n = n

    def getInitBoard(self):
        """获取初始棋盘状态
        Returns:
            numpy数组形式的初始棋盘
        """
        b = Board(self.n)
        return np.array(b.pieces)

    def getBoardSize(self):
        """获取棋盘尺寸
        Returns:
            (width, height)元组
        """
        return (self.n, self.n)

    def getActionSize(self):
        """获取动作空间大小
        Returns:
            可能的动作数(n*n个位置+1个pass动作)
        """
        return self.n*self.n + 1

    def getNextState(self, board, player, action):
        """执行动作后的下一个状态
        Args:
            board: 当前棋盘
            player: 当前玩家(1或-1)
            action: 执行的动作
        Returns:
            (next_board, next_player)元组
        """
        if action == self.n*self.n:  # pass动作
            return (board, -player)
        b = Board(self.n)
        b.pieces = np.copy(board)
        move = (int(action/self.n), action%self.n)
        b.execute_move(move, player)
        return (b.pieces, -player)

    def getValidMoves(self, board, player):
        """获取有效动作
        Args:
            board: 当前棋盘
            player: 当前玩家
        Returns:
            二进制向量,1表示合法动作
        """
        valids = [0]*self.getActionSize()
        b = Board(self.n)
        b.pieces = np.copy(board)
        legalMoves =  b.get_legal_moves(player)
        if len(legalMoves)==0:  # 无合法移动时pass
            valids[-1]=1
            return np.array(valids)
        for x, y in legalMoves:
            valids[self.n*x+y]=1
        return np.array(valids)

    def getGameEnded(self, board, player):
        """检查游戏是否结束
        Args:
            board: 当前棋盘
            player: 当前玩家
        Returns:
            None:未结束 1:player胜 -1:player负 0:平局
        """
        b = Board(self.n)
        b.pieces = np.copy(board)
        if b.has_legal_moves(player):
            return None
        if b.has_legal_moves(-player):
            return None
        if b.countDiff(player) > 0:
            return 1
        elif b.countDiff(player) < 0:
            return -1
        else:
            return 0

    def getCanonicalForm(self, board, player):
        """获取标准形式的状态
        Args:
            board: 当前棋盘
            player: 当前玩家
        Returns:
            player=1时返回原状态,=-1时返回相反状态
        """
        return player*board

    def getSymmetries(self, board, pi):
        """获取状态的对称形式
        Args:
            board: 当前棋盘
            pi: 策略向量
        Returns:
            所有旋转和翻转的等价状态列表
        """
        assert(len(pi) == self.n**2+1)  # pi长度应为n^2+1(含pass)
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []

        for i in range(1, 5):  # 4种旋转
            for j in [True, False]:  # 是否水平翻转
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l

    def stringRepresentation(self, board):
        """获取棋盘的字符串表示
        Args:
            board: 棋盘状态
        Returns:
            棋盘的二进制字符串表示
        """
        return board.tostring()

    def stringRepresentationReadable(self, board):
        """获取可读的棋盘字符串表示
        Args:
            board: 棋盘状态
        Returns:
            使用X/O/-表示的棋盘字符串
        """
        board_s = "".join(self.square_content[square] for row in board for square in row)
        return board_s

    def getScore(self, board, player):
        """获取玩家的得分
        Args:
            board: 当前棋盘
            player: 当前玩家
        Returns:
            player相对于对手的子数差
        """
        b = Board(self.n)
        b.pieces = np.copy(board)
        return b.countDiff(player)

    @staticmethod
    def display(board):
        """显示棋盘
        Args:
            board: 要显示的棋盘状态
        """
        n = board.shape[0]
        print("   ", end="")
        for y in range(n):
            print(y, end=" ")
        print("")
        print("-----------------------")
        for y in range(n):
            print(y, "|", end="")
            for x in range(n):
                piece = board[y][x]
                print(OthelloGame.square_content[piece], end=" ")
            print("|")

        print("-----------------------")


class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a

class GreedyOthelloPlayer():
    """贪心黑白棋玩家,每次选择能获得最大分数的移动"""
    def __init__(self, game):
        """初始化贪心玩家
        Args:
            game: 游戏实例
        """
        self.game = game

    def play(self, board):
        """选择移动
        Args:
            board: 当前棋盘状态
        Returns:
            选择的移动(整数)
        """
        valids = self.game.getValidMoves(board, 1)  # 获取有效移动
        candidates = []  # 存储(分数,移动)对的列表
        for a in range(self.game.getActionSize()):  # 遍历所有可能的移动
            if valids[a]==0:  # 跳过无效移动
                continue
            nextBoard, _ = self.game.getNextState(board, 1, a)  # 获取移动后的状态
            score = self.game.getScore(nextBoard, 1)  # 计算移动后的分数
            candidates += [(-score, a)]  # 添加(负分数,移动)对,使用负分数便于排序
        candidates.sort()  # 按分数排序
        return candidates[0][1]  # 返回得分最高的移动


class HumanOthelloPlayer():
    """人类黑白棋玩家,通过命令行输入移动"""
    def __init__(self, game):
        """初始化人类玩家
        Args:
            game: 游戏实例
        """
        self.game = game

    def play(self, board):
        """获取人类玩家的移动
        Args:
            board: 当前棋盘状态
        Returns:
            选择的移动(整数)
        """
        valid = self.game.getValidMoves(board, 1)  # 获取有效移动
        # 显示所有有效移动的坐标
        for i in range(len(valid)):
            if valid[i]:
                print("[", int(i/self.game.n), int(i%self.game.n), end="] ")
        # 循环直到获得有效输入
        while True:
            input_move = input()  # 获取用户输入
            input_a = input_move.split(" ")  # 分割输入的坐标
            if len(input_a) == 2:  # 如果输入了两个数
                try:
                    x,y = [int(i) for i in input_a]  # 转换为整数
                    # 检查坐标是否在有效范围内
                    if ((0 <= x) and (x < self.game.n) and (0 <= y) and (y < self.game.n)) or \
                            ((x == self.game.n) and (y == 0)):
                        a = self.game.n * x + y  # 转换为一维索引
                        if valid[a]:  # 如果是有效移动
                            break
                except ValueError:
                    'Invalid integer'  # 输入的不是整数
            print('Invalid move')  # 提示无效移动
        return a


class Arena():
    """对弈场景类,用于让两个玩家进行对弈"""

    def __init__(self, player1, player2, game, display=None):
        """初始化对弈场景
        Args:
            player1: 玩家1的移动函数
            player2: 玩家2的移动函数
            game: 游戏实例
            display: 显示棋盘的函数(用于详细模式)
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def playGame(self, verbose=False):
        """执行一局游戏
        Args:
            verbose: 是否显示详细信息
        Returns:
            winner: 获胜玩家(1表示玩家1,-1表示玩家2,0表示平局)
        """
        players = [self.player2, None, self.player1]  # 玩家列表,索引对应玩家编号(-1,0,1)
        curPlayer = 1  # 玩家1先手
        board = self.game.getInitBoard()  # 获取初始棋盘
        it = 0  # 回合计数
        while self.game.getGameEnded(board, curPlayer) is None:  # 游戏未结束时循环
            it += 1
            if verbose:  # 如果需要显示详细信息
                assert self.display
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(board)
            # 获取当前玩家的移动
            action = players[curPlayer + 1](self.game.getCanonicalForm(board, curPlayer))

            # 验证移动的有效性
            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)
            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0
            # 执行移动并更新状态
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
        # 计算游戏结果
        result = curPlayer * self.game.getGameEnded(board, curPlayer)
        if verbose:  # 显示最终结果
            assert self.display
            print("Game over: Turn ", str(it), "Result ", str(result))
            self.display(board)
        return result

    def playGames(self, num, verbose=False):
        """进行多局游戏
        每个玩家轮流先手,各进行num/2局
        Args:
            num: 总局数
            verbose: 是否显示详细信息
        Returns:
            oneWon: 玩家1获胜局数
            twoWon: 玩家2获胜局数
            draws: 平局局数
        """
        num = int(num / 2)  # 每个玩家先手的局数
        oneWon = 0  # 玩家1获胜次数
        twoWon = 0  # 玩家2获胜次数
        draws = 0   # 平局次数
        
        # 玩家1先手的对局
        for _ in tqdm(range(num), desc="Arena.playGames (player1 go first)"):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1

        # 交换玩家顺序
        self.player1, self.player2 = self.player2, self.player1

        # 玩家2先手的对局
        for _ in tqdm(range(num), desc="Arena.playGames (player2 go first)"):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == -1:
                oneWon += 1
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1

        return oneWon, twoWon, draws

