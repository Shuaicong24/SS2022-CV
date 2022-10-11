import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentLoss(nn.Module):
    """
    See Gatys et al. for the details.
    """

    def __init__(self, target, alpha):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.alpha = alpha

    def forward(self, input):
        self.loss = self.alpha * F.mse_loss(input, self.target)
        return input


def get_cosine_distance(x, y):
    # mean shifting by channel-wise mean of `y` = target.
    y_mu = y.mean(dim=(0, 2, 3), keepdim=True)
    x_centered = x - y_mu
    y_centered = y - y_mu

    # L2 normalization
    x_normalized = F.normalize(x_centered, p=2, dim=1)
    y_normalized = F.normalize(y_centered, p=2, dim=1)

    # channel-wise vectorization
    N, C, *_ = x.size()
    x_normalized = x_normalized.reshape(N, C, -1)  # (N, C, H*W)
    y_normalized = y_normalized.reshape(N, C, -1)  # (N, C, H*W)

    # consine similarity
    cosine_sim = torch.bmm(x_normalized.transpose(1, 2),
                           y_normalized)  # (N, H*W, H*W)

    # convert to distance
    return 1 - cosine_sim


class ContentLossDOS(nn.Module):
    def __init__(self, target, alpha2, alpha11):
        super(ContentLossDOS, self).__init__()
        self.target = target.detach()
        self.epsilon = 1e-5
        self.h = 0.5
        self.alpha2 = alpha2
        self.alpha11 = alpha11

    def get_cx_loss(self, input):
        distances = get_cosine_distance(input, self.target)
        minimums, _ = torch.min(distances, dim=2, keepdim=True)
        relative_dist = distances / (minimums + self.epsilon)

        similarities = torch.exp((1 - relative_dist) / self.h)
        contextual_similarities = similarities / torch.sum(similarities, dim=2, keepdim=True)

        max_CX, _ = torch.max(contextual_similarities, dim=1)
        CX = torch.mean(max_CX, dim=1)
        return torch.mean(-torch.log(CX + 1e-5))

    def forward(self, input):
        CX_loss = self.get_cx_loss(input)
        content_loss = F.mse_loss(input, self.target)
        self.loss = self.alpha2 * CX_loss + self.alpha11 * content_loss
        return input


def gram_matrix(input):
    B, C, H, W = input.size()
    features = input.view(B * C, H * W)
    gram = torch.mm(features, features.t())

    return gram.div(B * C * H * W)


class StyleLoss(nn.Module):
    """
    See Gatys et al. for the details.
    """

    def __init__(self, target_feature, beta):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.beta = beta

    def forward(self, input):
        gram = gram_matrix(input)
        self.loss = self.beta * F.mse_loss(gram, self.target)
        return input


class AugmentedStyleLoss(nn.Module):
    """
    AugmentedStyleLoss exploits the semantic information of images.
    See Luan et al. for the details.
    """

    def __init__(self, target_feature, target_masks, input_masks, alpha12):
        super(AugmentedStyleLoss, self).__init__()
        self.input_masks = [mask.detach() for mask in input_masks]
        self.targets = [
            gram_matrix(target_feature * mask).detach() for mask in target_masks
        ]
        self.alpha12 = alpha12

    def forward(self, input):
        gram_matrices = [
            gram_matrix(input * mask.detach()) for mask in self.input_masks
        ]
        self.loss = self.alpha12 * sum(
            F.mse_loss(gram, target)
            for gram, target in zip(gram_matrices, self.targets)
        )
        return input