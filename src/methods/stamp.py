import math
import random
from copy import deepcopy

import torch.nn as nn
import torch
from src.utils.utils import softmax_entropy
import torch.nn.functional as F
import torch.optim as optim
import logging
logger = logging.getLogger(__name__)


def entropy(p):
    return -torch.sum(p * torch.log(p + 1e-5), dim=1)


class RBM:
    def __init__(self, max_len, num_class):
        self.num_class = num_class
        self.count_class = torch.zeros(num_class)
        self.data = [[] for _ in range(num_class)]
        self.max_len = max_len
        self.total_num = 0

    def remove_item(self):
        max_count = 0
        for i in range(self.num_class):
            if len(self.data[i]) == 0:
                continue
            if self.count_class[i] > max_count:
                max_count = self.count_class[i]
        max_classes = []
        for i in range(self.num_class):
            if self.count_class[i] == max_count and len(self.data[i]) > 0:
                max_classes.append(i)
        remove_class = random.choice(max_classes)
        self.data[remove_class].pop(0)
        self.total_num -= 1  # 减少总数

    def append(self, items, class_ids):
        for item, class_id in zip(items, class_ids):
            if self.total_num < self.max_len:
                self.data[class_id].append(item)
                self.total_num += 1
            else:
                self.remove_item()
                self.data[class_id].append(item)
                self.total_num += 1  # 增加总数

    def get_data(self):
        data = []
        for cls in range(self.num_class):
            data.extend(self.data[cls])
            self.count_class[cls] = 0.9 * self.count_class[cls] + 0.1 * len(self.data[cls])
        if len(data) == 0:
            return torch.empty(0)
        return torch.stack(data)

    def __len__(self):
        return self.total_num

    def reset(self):
        self.count_class = torch.zeros(self.num_class)
        self.data = [[] for _ in range(self.num_class)]
        self.total_num = 0

    def replace_samples(self, indices, new_samples):
        """
        替换指定索引的内存样本。
        indices: tensor of indices to replace
        new_samples: tensor of new samples
        """
        all_data = []
        for cls_data in self.data:
            all_data.extend(cls_data)
        for idx, new_sample in zip(indices, new_samples):
            class_id, sample_idx = self.get_class_and_sample_idx(idx.item())
            if class_id is not None and sample_idx is not None:
                self.data[class_id][sample_idx] = new_sample.cpu()

    def get_class_and_sample_idx(self, global_idx):
        """
        根据全局索引获取对应的 class_id 和 sample_idx。
        """
        cumulative = 0
        for cls_id, cls_data in enumerate(self.data):
            if cumulative + len(cls_data) > global_idx:
                return cls_id, global_idx - cumulative
            cumulative += len(cls_data)
        return None, None


class MemoryEditor(nn.Module):
    def __init__(self, model, mem, edit_rate=1.0, editing_lr=0.01, device='cuda'):
        super(MemoryEditor, self).__init__()
        self.model = deepcopy(model).to(device)
        self.model.eval()  # 确保模型不在训练模式
        self.mem = mem
        self.edit_rate = edit_rate
        self.editing_lr = editing_lr  # 新增编辑学习率
        self.device = device
        self.edit_enabled = edit_rate > 0.0 and editing_lr > 0.0  # 同时检查两个参数
        logger.info(f"MemoryEditor initialized with edit_rate={self.edit_rate}, editing_lr={self.editing_lr}")

    def forward(self, update_memory_samples):
        if not self.edit_enabled or len(self.mem) == 0:
            logger.info("Memory editing is disabled or memory is empty.")
            return

        # 内存模块第一步：随机选取内存样本
        mem_data = self.mem.get_data().to(self.device)
        logger.info(f"Fetched memory data with shape: {mem_data.shape}")
        if len(mem_data) == 0:
            logger.info("Memory data is empty after fetching.")
            return

        # 随机选择一定比例的内存样本进行编辑
        num_edit = max(int(len(mem_data) * self.edit_rate), 1)  # 至少编辑一个样本
        indices = torch.randperm(len(mem_data))[:num_edit]
        logger.info(f"Number of samples to edit: {num_edit}")
        Xm = mem_data[indices].detach().clone().requires_grad_(True)
        logger.info(f"Xm.requires_grad: {Xm.requires_grad}")

        # 内存模块第一步：通过模型获得熵值
        with torch.no_grad():
            output_orig = self.model(Xm)
            mem_entropy = entropy(F.softmax(output_orig, dim=1))
            logger.info(f"mem_entropy: {mem_entropy}")

        # 内存模块第二步：伪更新（不影响主模型）
        pseudo_model = deepcopy(self.model).to(self.device)
        pseudo_model.train()
        # 临时将所有参数设置为 requires_grad=True
        for param in pseudo_model.parameters():
            param.requires_grad = True

        pseudo_optimizer = optim.SGD(pseudo_model.parameters(), lr=0.01)

        # 确认 pseudo_model 的参数需要梯度
        for name, param in pseudo_model.named_parameters():
            if param.requires_grad:
                logger.info(f"Pseudo model parameter {name} requires grad.")
            else:
                logger.info(f"Pseudo model parameter {name} does not require grad.")

        # 计算伪更新损失（最小化熵）
        output_pseudo = pseudo_model(update_memory_samples)
        logger.info(f"output_pseudo requires_grad: {output_pseudo.requires_grad}")
        logger.info(f"output_pseudo.grad_fn: {output_pseudo.grad_fn}")
        # 添加测试损失
        test_loss = output_pseudo.mean()
        logger.info(f"test_loss.requires_grad: {test_loss.requires_grad}")
        try:
            test_loss.backward()
            logger.info("Test loss backward successful.")
        except RuntimeError as e:
            logger.error(f"Test loss backward failed: {e}")

        loss_pseudo = softmax_entropy(output_pseudo).mean()
        logger.info(f"loss_pseudo requires_grad: {loss_pseudo.requires_grad}")

        pseudo_optimizer.zero_grad()
        try:
            loss_pseudo.backward()
            pseudo_optimizer.step()
        except RuntimeError as e:
            logger.error(f"loss_pseudo.backward() failed: {e}")
            raise e

        pseudo_optimizer.zero_grad()
        loss_pseudo.backward()
        pseudo_optimizer.step()

        # 内存模块第三步：通过伪更新后的模型获得新熵值
        with torch.no_grad():
            output_new = pseudo_model(Xm)
            new_mem_entropy = entropy(F.softmax(output_new, dim=1))
            logger.info(f"new_mem_entropy: {new_mem_entropy}")

        # 内存模块第四步：计算熵差
        diff_entropy = new_mem_entropy - mem_entropy
        logger.info(f"diff_entropy: {diff_entropy}")

        # 内存模块第五步：计算梯度并调整 Xm
        loss_edit = diff_entropy.mean()
        logger.info(f"loss_edit: {loss_edit}")
        loss_edit.backward()

        # 更新 Xm（梯度下降以最小化 diff_entropy）
        Xm_updated = Xm - self.editing_lr * Xm.grad
        Xm_updated = torch.clamp(Xm_updated, 0.0, 1.0)  # 假设输入在 [0,1] 之间
        logger.info(f"Xm_updated: {Xm_updated}")

        # 内存模块第六步：将编辑后的样本替换原有内存样本
        self.mem.replace_samples(indices, Xm_updated.detach())
        logger.info(f"Memory samples edited: {len(indices)} samples updated.")


class STAMP(nn.Module):
    def __init__(self, model, optimizer, alpha, num_class, edit_rate, editing_lr, device='cuda'):
        super(STAMP, self).__init__()
        self.model = self.configure_model(model)
        self.norm_model = deepcopy(self.model).train()
        self.num_class = num_class
        self.optimizer = optimizer
        self.alpha = alpha
        self.margin = alpha * math.log(num_class)
        self.mem = RBM(64, num_class)
        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)
        if num_class == 1000:
            self.max_iter = 750
        else:
            self.max_iter = 150
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.max_iter)
        
        # 初始化 MemoryEditor 并传递新的参数
        self.memory_editor = MemoryEditor(
            model=self.model,
            mem=self.mem,
            edit_rate=edit_rate,
            editing_lr=editing_lr,
            device=device
        )

    @staticmethod
    def configure_model(model):
        model.train()
        model.requires_grad_(False)
        # 配置 norm 进行 eata 更新：启用梯度 + 强制使用批量统计量
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # 强制在训练和评估模式下使用批量统计量
                m.track_running_stats = True
                # m.momentum = 0.2
                # m.running_mean = None
                # m.running_var = None
            if isinstance(m, nn.LayerNorm):
                m.requires_grad_(True)
        return model

    @staticmethod
    def collect_params(model):
        """
        收集批量归一化层的仿射缩放和偏移参数。
        """
        params = []
        names = []
        for nm, m in model.named_modules():
            # 跳过适应的顶层：ResNet 的 layer4 和 Vit-Base 的 blocks9-11
            if 'layer4' in nm:
                continue
            if 'conv5_x' in nm:
                continue
            if 'blocks.9' in nm:
                continue
            if 'blocks.10' in nm:
                continue
            if 'blocks.11' in nm:
                continue
            if 'norm.' in nm:
                continue
            if nm in ['norm']:
                continue

            if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight 是缩放，bias 是偏移
                        params.append(p)
                        names.append(f"{nm}.{np}")

        return params, names

    def forward(self, x):
        output = self.update_memory(x)
        
        # 在 adapt 之前进行内存编辑
        self.memory_editor(update_memory_samples=self.mem.get_data())
        
        if len(self.mem) != 0:
            self.adapt()
        return output

    def update_memory(self, x):
        x_origin = x[0]
        if self.num_class == 1000:
            outputs = []
            output_origin = self.model(x_origin)
            outputs.append(output_origin.softmax(dim=1))

            for i in range(1, len(x)):
                x_aug = x[i]
                outputs.append(self.model(x_aug).softmax(dim=1))
            output = torch.stack(outputs, dim=0)
            output = torch.mean(output, dim=0)

            entropys = entropy(output)
            filter_ids = torch.where(entropys < self.margin)
            x_append = x_origin[filter_ids]
            self.mem.append(x_append, output_origin.max(dim=1)[1][filter_ids])
        else:
            outputs = []
            self.model.train()
            output_origin = self.model(x_origin)
            output_norm = self.norm_model(x_origin)
            filter_ids_0 = torch.where(output_origin.max(dim=1)[1] == output_norm.max(dim=1)[1])
            outputs.append(output_origin.softmax(dim=1))
            for i in range(1, len(x)):
                x_aug = x[i]
                outputs.append(self.model(x_aug).softmax(dim=1))
            output = torch.stack(outputs, dim=0)
            output = torch.mean(output, dim=0)
            entropys = entropy(output)[filter_ids_0]
            filter_ids = torch.where(entropys < self.margin)
            x_append = x_origin[filter_ids_0][filter_ids]
            self.mem.append(x_append, output_origin.max(dim=1)[1][filter_ids_0][filter_ids])
        return output, -entropy(output)

    @torch.enable_grad()
    def adapt(self):
        data = self.mem.get_data()
        if len(data) == 0:
            return
        self.optimizer.zero_grad()
        if len(data) > 0:
            output_1 = self.model(data)
            entropys = softmax_entropy(output_1)
            # coeff = 1 / (torch.exp(entropys.clone().detach() - self.margin))
            inv_entropy = 1 / torch.exp(entropys)
            coeff = inv_entropy / inv_entropy.sum() * 64
            entropys = entropys.mul(coeff)
            loss = entropys.mean()
            loss.backward()
            self.optimizer.first_step(zero_grad=True)

            # 再次前向传播
            output_1 = self.model(data)
            entropys = softmax_entropy(output_1)
            inv_entropy = 1 / torch.exp(entropys)
            coeff = inv_entropy / inv_entropy.sum() * 64
            entropys = entropys.mul(coeff)
            loss = entropys.mean()
            loss.backward()
            self.optimizer.second_step(zero_grad=True)
            self.scheduler.step()

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)
        self.mem.reset()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.max_iter)


def copy_model_and_optimizer(model, optimizer):
    """复制模型和优化器的状态以便在适应后重置。"""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """从副本中恢复模型和优化器的状态。"""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


