import torch
import torch.nn as nn
import torch.optim as optim

import copy

from closed_form_matting import compute_laplacian
from data_utils import tensor_to_image, image_to_tensor
import losses


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


def get_style_model_and_losses(
    cnn,
    normalization_mean, normalization_std,
    style_img, content_img,
    style_masks, content_masks,
    device,
    alpha2, alpha11, alpha12,
    method,
    alpha, beta
):
    """
    Assumptions:
        - cnn is a nn.Sequential
        - resize happens only in the pooling layers
    """
    print('Building the style transfer model.')
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    style_layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
    content_layers = ["conv4_2"]

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    num_pool, num_conv = 0, 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            num_conv += 1
            name = "conv{}_{}".format(num_pool, num_conv)

        elif isinstance(layer, nn.ReLU):
            name = "relu{}_{}".format(num_pool, num_conv)
            layer = nn.ReLU(inplace=False)

        elif isinstance(layer, nn.MaxPool2d):
            num_pool += 1
            num_conv = 0
            name = "pool_{}".format(num_pool)
            if method == 'deepobjstyle':
                layer = nn.AvgPool2d(
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                )

                # Update the segmentation masks to match
                # the activation matrices of the neural responses.
                style_masks = [layer(mask) for mask in style_masks]
                content_masks = [layer(mask) for mask in content_masks]

        elif isinstance(layer, nn.BatchNorm2d):
            name = "bn{}_{}".format(num_pool, num_conv)

        else:
            raise RuntimeError(
                "Unrecognized layer: {}".format(layer.__class__.__name__)
            )

        model.add_module(name, layer)

        if method == 'deepobjstyle':
            if name in content_layers:
                target = model(content_img).detach()
                content_loss = losses.ContentLossDOS(target, alpha11=alpha11, alpha2=alpha2)
                model.add_module("content_loss_{}".format(num_pool), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                target_feature = model(style_img).detach()
                style_loss = losses.AugmentedStyleLoss(target_feature, style_masks, content_masks, alpha12=alpha12)
                model.add_module("style_loss_{}".format(num_pool), style_loss)
                style_losses.append(style_loss)

        if method == 'neural_style':
            if name in content_layers:
                target = model(content_img).detach()
                content_loss = losses.ContentLoss(target, alpha=alpha)
                model.add_module("content_loss_{}".format(num_pool), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                target_feature = model(style_img).detach()
                style_loss = losses.StyleLoss(target_feature, beta=beta)
                model.add_module("style_loss_{}".format(num_pool), style_loss)
                style_losses.append(style_loss)


    # Trim off the layers after the last content and style losses
    # to speed up forward pass.
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], (losses.ContentLoss, losses.ContentLossDOS, losses.StyleLoss, losses.AugmentedStyleLoss)):
            break

    model = model[: (i + 1)]

    return model, style_losses, content_losses


def run_style_transfer(
    cnn,
    normalization_mean, normalization_std,
    style_img, content_img, input_img,
    style_masks, content_masks,
    device,
    reg,
    num_steps,
    alpha2, alpha11, alpha12, alpha13,
    method,
    alpha, beta,
    style_layer_weight=0.2
):

    if method == 'deepobjstyle':
        run_style_transfer_dos(cnn, normalization_mean, normalization_std,
            style_img, content_img, input_img,
            style_masks, content_masks,
            device,
            reg,
            num_steps,
            alpha2, alpha11, alpha12, alpha13,
            method,
            alpha, beta,
            style_layer_weight)

    if method == 'neural_style':
        run_style_transfer_ns(cnn, normalization_mean, normalization_std,
            style_img, content_img, input_img,
            style_masks, content_masks,
            device,
            num_steps,
            alpha2, alpha11, alpha12,
            method,
            alpha, beta,
            style_layer_weight)

def run_style_transfer_dos(
    cnn,
    normalization_mean, normalization_std,
    style_img, content_img, input_img,
    style_masks, content_masks,
    device,
    reg,
    num_steps,
    alpha2, alpha11, alpha12, alpha13,
    method,
    alpha, beta,
    style_layer_weight
): 
    """
    Run the style transfer.
    `reg_weight` is the photorealistic regularization hyperparameter 
    """
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn,
        normalization_mean, normalization_std,
        style_img, content_img,
        style_masks, content_masks,
        device,
        alpha2, alpha11, alpha12,
        method,
        alpha, beta
    )
    optimizer = optim.LBFGS([input_img.requires_grad_()])

    if reg:
        L = compute_laplacian(tensor_to_image(content_img))

        def regularization_grad(input_img):
            """
            Photorealistic regularization
            See Luan et al. for the details.
            """
            im = tensor_to_image(input_img)
            grad = L.dot(im.reshape(-1, 3))
            loss = (grad * im.reshape(-1, 3)).sum()
            return loss, 2. * grad.reshape(*im.shape)

    print('Optimizing.')
    step = 0
    while step <= num_steps:

        def closure():
            """
            https://pytorch.org/docs/stable/optim.html#optimizer-step-closure
            """
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)

            get_loss = lambda x: x.loss
            style_score = style_layer_weight * sum(map(get_loss, style_losses))
            content_score = sum(map(get_loss, content_losses))

            loss = style_score + content_score
            loss.backward()

            # Add photorealistic regularization
            if reg:
                reg_loss, reg_grad = regularization_grad(input_img)
                reg_grad_tensor = image_to_tensor(reg_grad)

                input_img.grad += alpha13 * reg_grad_tensor.to(device)

                loss += alpha13 * reg_loss

            nonlocal step
            step += 1

            if step % 50 == 0:
                print(
                    "step {:>4d}:".format(step),
                    "S: {:.3f} C: {:.3f} R:{:.3f}".format(
                        style_score.item(), content_score.item(), alpha13 * reg_loss if reg else 0
                    ),
                )

            return loss

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)

    return input_img

def run_style_transfer_ns(
    cnn, 
    normalization_mean, normalization_std,
    style_img, content_img, input_img,
    style_masks, content_masks,
    device,
    num_steps,
    alpha2, alpha11, alpha12,
    method,
    alpha, beta,
    style_layer_weight
):
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn,
        normalization_mean, normalization_std,
        style_img, content_img,
        style_masks, content_masks,
        device,
        alpha2, alpha11, alpha12,
        method,
        alpha, beta
    )
        
    # We want to optimize the input and not the model parameters so we
    # update all the requires_grad fields accordingly
    input_img.requires_grad_(True)
    model.requires_grad_(False)

    optimizer = optim.LBFGS([input_img])

    style_scores = []
    content_scores = []

    print('Optimizing.')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += style_layer_weight * sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_scores.append(style_score.item())
            content_scores.append(content_score.item())

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}: Style Loss: {:4f} Content Loss: {:4f} ".format(run, style_score.item(), content_score.item()))

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img, style_scores, content_scores
