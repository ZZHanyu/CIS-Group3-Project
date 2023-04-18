import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
# from util import Metric
from model import Model
import scipy.sparse as sp
import math
from torch.autograd import grad
from UltraGCN import UltraGCNNet
from  UltraGCN_ERM import ERMNet


def setup_seed(seed):   # 随机数生成
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


class FrontModel(torch.nn.Module):
    # FrontModel承担 划分环境 的作用
    # 首先定义并初始化训练用神经网络 UltraGCN的结构和参数
    # 并初始化训练参数
    def __init__(self, ds, args, logging):
        super().__init__() # 调用父类
        setup_seed(2233)
        # 定义+调用神经网络的参数
        self.ds = ds
        self.args = args
        self.logging = logging
        self.filename_pre = 'weights/%s_UGCN_best.pth' % args.dataset # .pth文件通过有序字典来保持模型参数

        self.net = UltraGCNNet(self.ds, self.args, self.logging).to(self.args.device)
        # 每一个FrontModel的实例都有一个network，每个UltraGCNnet都存储自己的训练经验

        self.weight = None

    def predict(self, uid, iid, flag=False):
        return self.net.predict(uid, iid, flag)

    def reg_loss(self):
        lr2, wd2 = self.args.p_proj
        loss = torch.mean(torch.abs(self.net.MLP.weight * self.weight))
        return wd2 * loss

    def init_frontmodel(self):
        self.net.load_state_dict(torch.load(self.filename_pre), strict=False) # 从保存的模型中加载数据
        for p in self.net.parameters(): # 对所有的参数禁用梯度
            p.requires_grad = False
        torch.nn.init.normal_(self.net.MLP.weight, mean=0.0, std=0.01)
        torch.nn.init.constant_(self.net.MLP.bias, 0)
        self.net.MLP.weight.requires_grad = True
        self.net.MLP.bias.requires_grad = True

    def train(self, m_weight, domain, current_domain):
        self.weight = m_weight
        lr1, wd1 = self.args.p_emb
        lr2, wd2 = self.args.p_proj
        # 自适应梯度下降优化器
        optimizer = torch.optim.Adagrad(self.net.emb_params, lr=lr1, weight_decay=0)
        optimizer2 = torch.optim.Adam(self.net.proj_params, lr=lr2, weight_decay=0)

        epochs = self.args.f_epoch
        # 下面正式开始训练：
        for epoch in tqdm(range(epochs)): # 进度条插件
            generator = self.ds.sample(domain, current_domain)
            # generator是数据集，
            # sample 从domain这个列表只能够随即抽取 current_domain个函数

            loss_sum = 0.0
            while True:
                self.net.train()
                optimizer.zero_grad()  # 将模型的参数梯度初始化为0
                optimizer2.zero_grad()
                uid, iid, niid = next(generator)
                # 加载下一条用户数据
                if uid is None:
                    break
                uid, iid, niid = uid.to(self.args.device), iid.to(self.args.device), niid.to(self.args.device)
                # user identify / item 特征
                loss = self.net(uid, iid, niid) + self.reg_loss()
                loss_sum += loss.detach()
                # 损失函数

                loss.backward()
                # 通过损失向后更新参数

                optimizer.step()
                # 更新所有参数
                optimizer2.step()

            if epoch > 0 and (epoch + 1) % self.args.epoch == 0:
                self.logging.info("Epoch %d: loss %s, U.norm %s, V.norm %s, MLP.norm %s" % (epoch + 1, loss_sum, torch.norm(self.net.U).item(), torch.norm(self.net.V).item(), torch.norm(self.net.MLP.weight).item()))


class FeatureSelector(torch.nn.Module):
    def __init__(self, input_dim, sigma, args):
        super().__init__()
        setup_seed(2233)
        self.args = args
        # self.mu 记录每一张图片的特征，这里初始化特征矩阵mu为0:
        self.mu = torch.nn.Parameter(torch.zeros(input_dim), requires_grad=True)
        self.noise = torch.randn(self.mu.size()).to(self.args.device)
        # 随机噪声序列
        self.sigma = sigma
        self.input_dim = input_dim

    def renew(self):
        torch.nn.init.zeros_(self.mu)
        self.noise = torch.randn(self.mu.size()).to(self.args.device)
        # .to()这行代码的意思是将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行。

    def forward(self, prev_x):
        z = self.mu + self.sigma * self.noise.normal_() * self.training
        stochastic_gate = self.hard_sigmoid(z)
        new_x = prev_x * stochastic_gate
        return new_x

    def hard_sigmoid(self, x):
        return torch.clamp(x + 0.5, 0.0, 1.0)

    def reg(self, x):
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


class InvRL(Model):
    def __init__(self, ds, args, logging):
        super().__init__()
        setup_seed(2233)
        self.filename_pre = 'weights/%s_UGCN_best.pth' % args.dataset
        self.filename = 'weights/%s_InvRL_best.pth' % args.dataset
        self.ds = ds
        self.args = args
        self.logging = logging

        self.max_test = None
        self.max_net = None

        self.mask_dim = self.ds.feature.shape[1]
        self.domain = torch.tensor(np.random.randint(0, self.args.num_domains, int(self.ds.train.shape[0]*0.01))).to(self.args.device)
        self.weight = torch.tensor(np.zeros(self.mask_dim, dtype=np.float32)).to(self.args.device)
        self.proj = None

        self.net_e = None

        # 第一步： 提取特征
        self.fs = FeatureSelector(self.mask_dim, self.args.sigma, args).to(self.args.device)
        self.lam = self.args.lam
        self.alpha = self.args.alpha
        # backmodel 定义在此
        self.backmodel = UltraGCNNet(self.ds, self.args, self.logging, has_bias=False).to(self.args.device)

        self.net = None

    def cal_error(self, net_id):
        with torch.no_grad():
            result = torch.Tensor([]).to(self.args.device)
            start_index = 0
            end_index = self.args.ssz
            while start_index < end_index <= self.ds.sz:
                sub = torch.from_numpy(self.ds.train[start_index: end_index, :]).to(self.args.device)
                pred = self.net_e.predict(sub[:, 0], sub[:, 1])
                if pred is None:
                    pred = torch.zeros_like(sub[:, 0])
                result = torch.cat((result, pred), dim=0)
                start_index = end_index
                end_index += self.args.ssz
                if end_index >= self.ds.sz:
                    end_index = self.ds.sz
            return result

    def init_word_emb(self, net):
        word_emb = torch.load(self.filename_pre,map_location=torch.device('cpu'))['word_emb']
        net.word_emb.data.copy_(word_emb)
        net.word_emb.requires_grad = False

    def frontend(self):
        # 真正开始 划分环境
        self.logging.info('----- frontend -----')
        ite = 0
        delta_threshold = int(self.ds.train.shape[0] * 0.01)
        print('delta_threshold %d' % delta_threshold)
        if self.args.reuse == 0:
            self.domain = torch.tensor(np.random.randint(0, self.args.num_domains, int(self.ds.train.shape[0]*0.01))).to(
                self.args.device)
            print('domain :', self.domain)

        while True:
            ite += 1 # 迭代是 batch 需要完成一个 epoch 的次数。记住：在一个 epoch 中，batch 数和迭代数是相等的。
            self.net_e = None
            tot_result = []
            for i in range(self.args.num_domains):
                self.logging.info('Environment %d' % i)
                self.net_e = FrontModel(self.ds, self.args, self.logging).to(self.args.device)
                # environment network
                # 加载抖音训练数据集，并开始训练划分环境：
                if self.args.dataset == 'tiktok':
                    self.init_word_emb(self.net_e.net)
                self.net_e.train(self.weight, self.domain, i)
                result = self.cal_error(i)
                tot_result.append(result)    # 所有的预测结果 是一个集合 对应论文大T

            tot_result = torch.stack(tot_result, dim=0)
            new_domain = torch.argmax(tot_result, dim=0)    # 对应 论文(7)式
            diff = self.domain.reshape(-1, 1) - new_domain.reshape(-1, 1)
            diff[diff != 0] = 1
            delta = int(torch.sum(diff))
            print('Ite = %d, Delta = %d' % (ite, delta))
            self.logging.info('Ite = %d, Delta = %d' % (ite, delta))
            self.domain = new_domain
            if delta <= delta_threshold or ite >= self.args.f_max:
                break

        print(self.domain)
        self.net_e = None

    def predict(self, uid, iid, flag=False):
        return self.net.predict(uid, iid, flag)

    def single_forward(self, uid, iid, niid):   # 单次前进，得到单次训练的Loss和Grad（梯度）
        assert self.fs.training is True
        loss_single = self.backmodel(uid, iid, niid, self.fs)
        grad_single = grad(loss_single, self.backmodel.MLP.weight, create_graph=True)[0]
        return loss_single, grad_single

    def loss_p(self, loss_avg, grad_avg, grad_list):    # 这里对应 mm论文中（2）式，也就是IRM计算过程
        penalty = torch.zeros_like(grad_avg).to(self.args.device)
        for gradient in grad_list:
            penalty += (gradient - grad_avg) ** 2
        penalty_detach = torch.sum((penalty * (self.fs.mu + 0.5)) ** 2)
        reg = self.fs.reg((self.fs.mu + 0.5) / self.fs.sigma) # 正则化
        reg_penalty = torch.sum(self.fs.mu ** 2)
        total_loss = loss_avg + self.alpha * penalty_detach
        total_loss = total_loss + self.lam * reg_penalty
        return total_loss, penalty_detach, reg

    def init_backmodel(self):
        self.backmodel.load_state_dict(torch.load(self.filename_pre), strict=False)
        self.fs.renew()
        for p in self.backmodel.parameters():
            p.requires_grad = False
        torch.nn.init.normal_(self.backmodel.MLP.weight, mean=0.0, std=0.01)
        self.backmodel.MLP.weight.requires_grad = True


    def backend(self):
        # BackModel是FrontModel划分环境之后，在Variant Representations基础上对Mask进行学习的过程
        # 所学到的 Mask (m) 被用于后续FrontModel 环境划分
        # 如果我们需要在划分环境之后应用ERM，ERM应该添加至此处

        # IRM 是对不同环境下不变特征的学习，学习数据集必须是不同环境
        # ERM 经验风险最小化，ERM可以在不同环境和相同环境下学习，普适性更强

        self.logging.info('----- backend -----')
        self.init_backmodel() # 初始化 backmodel中的参数
        lr1, wd1 = self.args.p_emb
        lr2, wd2 = self.args.p_proj
        # Adam 一种自适应调节 学习率/梯度/步长etc 的机器学习优化方法
        optimizer2 = torch.optim.Adam([{'params': self.backmodel.proj_params, 'lr': lr2, 'weight_decay': 0}, {'params': self.fs.mu, 'lr': self.args.lr, 'weight_decay': 0}])

        epochs = self.args.b_epoch
        reg = None

        for epoch in tqdm(range(epochs)):
            generator = []
            for i in range(self.args.num_domains):
                generator.append(self.ds.sample(self.domain, i))
            end_flag = False
            finish = [0 for i in range(self.args.num_domains)]
            loss = 0.0
            while end_flag is False:
                self.backmodel.train()
                self.fs.train()
                optimizer2.zero_grad()
                loss_avg = 0.0
                grad_avg = torch.zeros_like(self.backmodel.MLP.weight).to(self.args.device)  # 0.0
                grad_list = []
                for i in range(self.args.num_domains):
                    uid, iid, niid = next(generator[i]) # next()返回下一个迭代器
                    if uid is None:
                        finish[i] = 1
                        if sum(finish) < self.args.num_domains:
                            generator[i] = self.ds.sample(self.domain, i)
                            uid, iid, niid = next(generator[i])
                        else:
                            end_flag = True
                            break
                    if uid is None:
                        continue
                    # 将特征发送到GPU进行训练
                    uid, iid, niid = uid.to(self.args.device), iid.to(self.args.device), niid.to(self.args.device)
                    loss_single, grad_single = self.single_forward(uid, iid, niid)
                    assert loss_single >= 0 # 判断为False激活
                    loss_avg += loss_single / self.args.num_domains
                    grad_avg += grad_single / self.args.num_domains
                    grad_list.append(grad_single)

                loss, penalty, reg = self.loss_p(loss_avg, grad_avg, grad_list)
                loss.backward() # Computes the gradient of current tensor w.r.t. graph leaves.
                optimizer2.step() # 更新全部参数

            if epoch > 0 and (epoch + 1) % self.args.epoch == 0: # 当一个完整的数据集通过了神经网络一次并且返回了一次，这个过程称为一个 epoch。
                self.logging.info("Epoch %d: loss %s, reg %s mu %s MLP.norm %s" % (epoch + 1, loss, reg, self.fs.mu, torch.norm(self.backmodel.MLP.weight)))

        self.proj = self.backmodel.MLP.weight.detach()

        return self.fs.hard_sigmoid(self.fs.mu).detach(), reg.detach()


    def solve(self, ite=3):
        #   训练mask
        #   其中，weight即为mask
        for i in range(ite):
            self.frontend()
            weight, density = self.backend()
            self.weight = weight
            self.lam *= 1.05
            self.alpha *= 1.05

        self.backmodel = None


    def train_erm(self):
        if self.args.pretrained == 0:
            self.solve(self.args.ite)
            mask = self.weight
        else:
            mask = np.load(self.mask_filename, allow_pickle=True)
            mask = torch.from_numpy(mask)
        self.logging.info('mask %s' % mask)

        #   取invariant representaion的反集：
        variant_representation = torch.ones(mask.shape) - mask
        print(variant_representation)

        #   定义模型和参数
        self.args.p_emb = self.args.p_embp
        self.args.p_proj = self.args.p_ctx
        self.net = ERMNet(self.ds, self.args, self.logging, mask.cpu()).to(self.args.device)

        #   定义数据集
        if self.args.dataset == 'tiktok':
            self.init_word_emb(self.net)

        #   定义学习率：
        lr1, wd1 = self.args.p_emb
        lr2, wd2 = self.args.p_proj
        optimizer = torch.optim.Adam(self.net.emb_params, lr=lr1, weight_decay=0)
        optimizer2 = torch.optim.Adam(self.net.proj_params, lr=lr2, weight_decay=0)


        #   初始化参数：
        epochs = self.args.num_epoch
        val_max = 0.0
        num_decreases = 0
        max_epoch = 0
        end_epoch = epochs
        loss = 0.0
        self.fs.eval()
        assert self.fs.training is False


        #   将变化环境抽离出来进行单独的ERM学习：
        for epoch in tqdm(range(epochs)):
            generator = self.ds.sample()
            while True:
                self.net.train()
                optimizer.zero_grad()
                optimizer2.zero_grad()
                uid, iid, niid = next(generator)
                if uid is None:
                    break
                uid, iid, niid = uid.to(self.args.device), iid.to(self.args.device), niid.to(self.args.device)

                loss = self.net(uid, iid, niid, mask)

                loss.backward()  # Computes the gradient of current tensor w.r.t. graph leaves.
                optimizer.step()
                optimizer2.step()

            if epoch > 0 and epoch % self.args.epoch == 0:
                self.logging.info("Epoch %d: loss %s, U.norm %s, V.norm %s, MLP.norm %s" % (
                epoch, loss, torch.norm(self.net.U).item(), torch.norm(self.net.V).item(),
                torch.norm(self.net.MLP.weight).item()))
                self.val(), self.test()
                if self.val_ndcg > val_max:  # 冒泡法
                    val_max = self.val_ndcg
                    max_epoch = epoch
                    num_decreases = 0
                    self.update()
                else:
                    if num_decreases > 40:
                        end_epoch = epoch
                        break
                    else:
                        num_decreases += 1

        self.logging.info("Epoch %d:" % end_epoch)
        self.val(), self.test()
        if self.val_ndcg > val_max:
            val_max = self.val_ndcg
            max_epoch = epochs
            num_decreases = 0
            self.update()

def train(self):
        if self.args.pretrained == 0:
            self.solve(self.args.ite)
            mask = self.weight
        else:
            mask = np.load(self.mask_filename, allow_pickle=True)
            mask = torch.from_numpy(mask)
        self.logging.info('mask %s' % mask)


        self.args.p_emb = self.args.p_embp
        self.args.p_proj = self.args.p_ctx
        self.net = UltraGCNNet(self.ds, self.args, self.logging, mask.cpu()).to(self.args.device)

        if self.args.dataset == 'tiktok':
            self.init_word_emb(self.net)

        lr1, wd1 = self.args.p_emb
        lr2, wd2 = self.args.p_proj
        optimizer = torch.optim.Adam(self.net.emb_params, lr=lr1, weight_decay=0)
        optimizer2 = torch.optim.Adam(self.net.proj_params, lr=lr2, weight_decay=0)

        epochs = self.args.num_epoch
        val_max = 0.0
        num_decreases = 0
        max_epoch = 0
        end_epoch = epochs
        loss = 0.0
        self.fs.eval()
        assert self.fs.training is False


        #从这里开始 IRM学习：
        for epoch in tqdm(range(epochs)):   # epoch 是完整跑完一边数据集，训练过程不只跑一遍数据
            #   这里的学习是对划分环境之后的学习
            generator = self.ds.sample()
            while True:
                self.net.train()
                optimizer.zero_grad()
                optimizer2.zero_grad()
                uid, iid, niid = next(generator)
                if uid is None:
                    break
                uid, iid, niid = uid.to(self.args.device), iid.to(self.args.device), niid.to(self.args.device)

                loss = self.net(uid, iid, niid)

                loss.backward()  # Computes the gradient of current tensor w.r.t. graph leaves.
                optimizer.step()
                optimizer2.step()

            if epoch > 0 and epoch % self.args.epoch == 0:
                self.logging.info("Epoch %d: loss %s, U.norm %s, V.norm %s, MLP.norm %s" % (epoch, loss, torch.norm(self.net.U).item(), torch.norm(self.net.V).item(), torch.norm(self.net.MLP.weight).item()))
                self.val(), self.test()
                if self.val_ndcg > val_max: # 冒泡法
                    val_max = self.val_ndcg
                    max_epoch = epoch
                    num_decreases = 0
                    self.update()
                else:
                    if num_decreases > 40:
                        end_epoch = epoch
                        break
                    else:
                        num_decreases += 1

        self.logging.info("Epoch %d:" % end_epoch)
        self.val(), self.test()
        if self.val_ndcg > val_max:
            val_max = self.val_ndcg
            max_epoch = epochs
            num_decreases = 0
            self.update()

        self.logging.info("final:")
        self.logging.info('----- test -----')
        self.logscore(self.max_test)
        self.logging.info('max_epoch %d:' % max_epoch)


