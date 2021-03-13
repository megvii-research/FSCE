import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init


class ContrastiveHead(nn.Module):
    """MLP head for contrastive representation learning, https://arxiv.org/abs/2003.04297
    Args:
        dim_in (int): dimension of the feature intended to be contrastively learned
        feat_dim (int): dim of the feature to calculated contrastive loss

    Return:
        feat_normalized (tensor): L-2 normalized encoded feature,
            so the cross-feature dot-product is cosine similarity (https://arxiv.org/abs/2004.11362)
    """
    def __init__(self, dim_in, feat_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, feat_dim),
        )
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        feat = self.head(x)
        feat_normalized = F.normalize(feat, dim=1)
        return feat_normalized


class SupConLoss(nn.Module):
    """Supervised Contrastive LOSS as defined in https://arxiv.org/pdf/2004.11362.pdf."""

    def __init__(self, temperature=0.2, iou_threshold=0.5, reweight_func='none'):
        '''Args:
            tempearture: a constant to be divided by consine similarity to enlarge the magnitude
            iou_threshold: consider proposals with higher credibility to increase consistency.
        '''
        super().__init__()
        self.temperature = temperature
        self.iou_threshold = iou_threshold
        self.reweight_func = reweight_func

    def forward(self, features, labels, ious):
        """
        Args:
            features (tensor): shape of [M, K] where M is the number of features to be compared,
                and K is the feature_dim.   e.g., [8192, 128]
            labels (tensor): shape of [M].  e.g., [8192]
        """
        assert features.shape[0] == labels.shape[0] == ious.shape[0]

        if len(labels.shape) == 1:
            labels = labels.reshape(-1, 1)

        # mask of shape [None, None], mask_{i, j}=1 if sample i and sample j have the same label
        label_mask = torch.eq(labels, labels.T).float().cuda()

        similarity = torch.div(
            torch.matmul(features, features.T), self.temperature)
        # for numerical stability
        sim_row_max, _ = torch.max(similarity, dim=1, keepdim=True)
        similarity = similarity - sim_row_max.detach()

        # mask out self-contrastive
        logits_mask = torch.ones_like(similarity)
        logits_mask.fill_diagonal_(0)

        exp_sim = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))

        per_label_log_prob = (log_prob * logits_mask * label_mask).sum(1) / label_mask.sum(1)

        keep = ious >= self.iou_threshold
        per_label_log_prob = per_label_log_prob[keep]
        loss = -per_label_log_prob

        coef = self._get_reweight_func(self.reweight_func)(ious)
        coef = coef[keep]

        loss = loss * coef
        return loss.mean()

    @staticmethod
    def _get_reweight_func(option):
        def trivial(iou):
            return torch.ones_like(iou)
        def exp_decay(iou):
            return torch.exp(iou) - 1
        def linear(iou):
            return iou

        if option == 'none':
            return trivial
        elif option == 'linear':
            return linear
        elif option == 'exp':
            return exp_decay


class SupConLossV2(nn.Module):
    def __init__(self, temperature=0.2, iou_threshold=0.5):
        super().__init__()
        self.temperature = temperature
        self.iou_threshold = iou_threshold

    def forward(self, features, labels, ious):
        if len(labels.shape) == 1:
            labels = labels.reshape(-1, 1)

        # mask of shape [None, None], mask_{i, j}=1 if sample i and sample j have the same label
        label_mask = torch.eq(labels, labels.T).float().cuda()

        similarity = torch.div(
            torch.matmul(features, features.T), self.temperature)
        # for numerical stability
        sim_row_max, _ = torch.max(similarity, dim=1, keepdim=True)
        similarity = similarity - sim_row_max.detach()

        # mask out self-contrastive
        logits_mask = torch.ones_like(similarity)
        logits_mask.fill_diagonal_(0)


        exp_sim = torch.exp(similarity)
        mask = logits_mask * label_mask
        keep = (mask.sum(1) != 0 ) & (ious >= self.iou_threshold)

        log_prob = torch.log(
            (exp_sim[keep] * mask[keep]).sum(1) / (exp_sim[keep] * logits_mask[keep]).sum(1)
        )

        loss = -log_prob
        return loss.mean()


class SupConLossWithStorage(nn.Module):
    def __init__(self, temperature=0.2, iou_threshold=0.5):
        super().__init__()
        self.temperature = temperature
        self.iou_threshold = iou_threshold

    def forward(self, features, labels, ious, queue, queue_label):
        fg = queue_label != -1
        # print('queue', torch.sum(fg))
        queue = queue[fg]
        queue_label = queue_label[fg]

        keep = ious >= self.iou_threshold
        features = features[keep]
        feat_extend = torch.cat([features, queue], dim=0)

        if len(labels.shape) == 1:
            labels = labels.reshape(-1, 1)
        labels = labels[keep]
        queue_label = queue_label.reshape(-1, 1)
        label_extend = torch.cat([labels, queue_label], dim=0)

        # mask of shape [None, None], mask_{i, j}=1 if sample i and sample j have the same label
        label_mask = torch.eq(labels, label_extend.T).float().cuda()

        # print('# companies', label_mask.sum(1))

        similarity = torch.div(
            torch.matmul(features, feat_extend.T), self.temperature)
        # print('logits range', similarity.max(), similarity.min())

        # for numerical stability
        sim_row_max, _ = torch.max(similarity, dim=1, keepdim=True)
        similarity = similarity - sim_row_max.detach()

        # mask out self-contrastive
        logits_mask = torch.ones_like(similarity)
        logits_mask.fill_diagonal_(0)

        exp_sim = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))

        per_label_log_prob = (log_prob * logits_mask * label_mask).sum(1) / label_mask.sum(1)
        loss = -per_label_log_prob
        return loss.mean()


class SupConLossWithPrototype(nn.Module):
    '''TODO'''

    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels, protos, proto_labels):
        """
        Args:
            features (tensor): shape of [M, K] where M is the number of features to be compared,
                and K is the feature_dim.   e.g., [8192, 128]
            labels (tensor): shape of [M].  e.g., [8192]
            proto (tensor): shape of [B, 128]
            proto_labels (tensor), shape of [B], where B is number of prototype (base) classes
        """
        assert features.shape[0] == labels.shape[0]
        fg_index = labels != self.num_classes

        features = features[fg_index]  # [m, 128]
        labels = labels[fg_index]      # [m, 128]
        numel = features.shape[0]      # m is named numel

        # m  =  n  +  b
        base_index = torch.eq(labels, proto_labels.reshape(-1,1)).any(axis=0)  # b
        novel_index = ~base_index  # n
        if torch.sum(novel_index) > 1:
            ni_pk = torch.div(torch.matmul(features[novel_index], protos.T), self.temperature)  # [n, B]
            ni_nj = torch.div(torch.matmul(features[novel_index], features[novel_index].T), self.temperature)  # [n, n]
            novel_numer_mask = torch.ones_like(ni_nj)  # mask out self-contrastive
            novel_numer_mask.fill_diagonal_(0)
            exp_ni_nj = torch.exp(ni_nj) * novel_numer_mask  # k != i
            novel_label_mask = torch.eq(labels[novel_index], labels[novel_index].T)
            novel_log_prob = ni_nj - torch.log(exp_ni_nj.sum(dim=1, keepdim=True) + ni_pk.sum(dim=1, keepdim=True))
            loss_novel = -(novel_log_prob * novel_numer_mask * novel_label_mask).sum(1) / (novel_label_mask * novel_numer_mask).sum(1)
            loss_novel = loss_novel.sum()
        else:
            loss_novel = 0

        if torch.any(base_index):
            bi_pi = torch.div(torch.einsum('nc,nc->n', features[base_index], protos[labels[base_index]]), self.temperature) # shape = [b]
            bi_nk = torch.div(torch.matmul(features[base_index], features[novel_index].T), self.temperature)  # [b, n]
            bi_pk = torch.div(torch.matmul(features[base_index], protos.T), self.temperature)  # shape = [b, B]
            # bi_pk_mask = torch.ones_like(bi_pk)
            # bi_pk_mask.scatter_(1, labels[base_index].reshape(-1, 1), 0)
            # base_log_prob = bi_pi - torch.log(torch.exp(bi_nk).sum(1) + (torch.exp(bi_pk) * bi_pk_mask).sum(1))
            base_log_prob = bi_pi - torch.log(torch.exp(bi_nk).sum(1) + torch.exp(bi_pk).sum(1))
            loss_base = -base_log_prob
            loss_base = loss_base.sum()
        else:
            loss_base = 0

        loss = (loss_novel + loss_base) / numel
        try:
            assert loss >= 0
        except:
            print('novel', loss_novel)
            print('base', loss_base)
            exit('loss become negative.')
        return loss