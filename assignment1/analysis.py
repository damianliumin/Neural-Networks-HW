# Created by LIU Min
# 191240030@smail.nju.edu.cn

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pylab import subplots_adjust
from lime import lime_image
from skimage.segmentation import mark_boundaries

from model.ModelManager import load_checkpoint
from data import prep_dataloader


def main():
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    model = load_checkpoint('checkpoints/rsn11_checkpoint_best.pt')
    model = model.to(device)

    dataset = prep_dataloader('AS1_data/train.csv', mode='analysis', num_workers=2, batchsz=1)

    # mkdir
    if not os.path.exists('analyses'):
        os.mkdir('analyses')
    for folder in ('gradient_ascent', 'lime', 'saliency'):
        if not os.path.exists(f'analyses/{folder}'):
            os.mkdir(f'analyses/{folder}')

    # Saliency Maps
    # saliency_map(device, model, dataset, 50)

    # Gradient Ascent
    # plt.figure()
    # title = 'layer4_block1_conv2'
    # for i in range(6):
    #     img = gradient_ascent(device, model, model.resnet.layer4[1].conv2, i)
    #     plt.subplot(2, 3, i + 1)
    #     plt.axis('off')
    #     plt.imshow(img.detach().cpu().numpy())
    # plt.tight_layout()
    # plt.suptitle(title)
    # plt.savefig(f'analyses/gradient_ascent/{title}.png', dpi=400)
    # plt.close()

    # Get avg outputs of filters
    # test_filter(device, model, dataset, model.resnet.layer4[0].conv1, model.resnet.layer4[0].conv1.out_channels)

    # Confusion Matrix
    # confusion_matrix(device, model, dataset)

    # LIME
    # for i in range(7):
    #     lime(device, model, dataset, i, False, 2, 1)
    #     lime(device, model, dataset, i, False, 2, 2)

    # filter out
    filter_out(device, model, dataset, 0)

def gradient_ascent(device, model, layer, idx):
    model.eval()
    # randomized image
    input = torch.rand(1, 3, 48, 48)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1).repeat(1, 48, 48)
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1).repeat(1, 48, 48)
    input = ((input - mean) / std).to(device)
    input.requires_grad_()

    # get hook
    filter_output = []
    def hook(module, input, output):
        filter_output.append(output[0, idx, :, :])
        return None
    layer.register_forward_hook(hook)
    
    def compute_loss():
        out = filter_output[-1][:, :]
        return torch.mean(out)

    def step(img, lr):
        img.retain_grad()
        _ = model(img)
        loss = compute_loss()
        print('loss: {:.6f}'.format(loss.item()), end=' ')
        loss.backward()
        print('grad: {:.9f}'.format(img.grad.mean().item()))
        return img + lr * img.grad.data

    # iteration
    iteration = 500
    lr = 150.0
    for _ in range(iteration):
        input = step(input, lr)

    # process img
    def process(img):
        img = torch.mean(img, dim=1).squeeze_()  # 48 * 48
        img -= img.min()
        img /= img.max()
        img *= 255
        return img
    return process(input)


def test_filter(device, model, dataset, layer, nfilter=6):
    model.eval()
    targets = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}

    rec = {i:[] for i in range(7)}  # rec[emotion_id][i]: size (nfilter)
    filter_out = []
    def hook(module, input, output):
        filter_out.append(torch.mean(output[0, :nfilter, :, :], dim=[1, 2]))
        return None
    layer.register_forward_hook(hook)

    for batch_idx, (x, y) in enumerate(dataset):
        x, y = x.to(device), y.to(device)
        out = model(x)
        y_pred = torch.argmax(F.softmax(out, dim=1), dim=1)
        if y != y_pred:
            continue
        rec[y.cpu().item()].append(filter_out[-1].cpu())

    rec_outs = []
    for i in range(7):
        outs = torch.mean(torch.stack(rec[i], 0), 0)
        rec_outs.append(outs)

    rec_outs = torch.stack(rec_outs, 0).transpose(0, 1)  # nfilter x 7
    for i in range(nfilter):
        print('+=============================+')
        print(f'filter {i}')
        for j in range(7):
            print(f'{targets[j]}: {rec_outs[i, j]}', 
                end=' *\n' if rec_outs[i, j] == rec_outs[i].max() else '\n')

    # statistics
    _, id = torch.max(rec_outs, dim=1)
    print(id, id.shape, sep='\n')
    for i in range(7):
        print('{}: {}'.format(targets[i], torch.where(id == i, 1, 0).sum()))


def filter_out(device, model, dataset, idx=0):
    model.eval()
    filter_out = []
    def hook(module, input, output):
        filter_out.append(output[0, :64, :, :])
        return None
    model.resnet.conv1.register_forward_hook(hook)

    for batch_idx, (x, y) in enumerate(dataset):
        if batch_idx != idx:
            continue
        x, y = x.to(device), y.to(device)
        out = model(x)
        break

    plt.figure()
    for i in range(64):
        img = filter_out[0][i]
        plt.subplot(8, 8, i+1)
        plt.axis('off')
        plt.imshow(img.detach().cpu().numpy(), cmap='pink')
    plt.tight_layout()
    plt.savefig(f'analyses/gradient_ascent/filter_out.png', dpi=400)
    plt.close()
    

def confusion_matrix(device, model, dataset):
    model.eval()

    cm = np.zeros((7, 7))
    for batch_idx, (x, y) in enumerate(dataset):
        x, y = x.to(device), y.to(device)
        out = model(x)
        y_pred = torch.argmax(F.softmax(out, dim=1), dim=1)
        cm[y.cpu().item(), y_pred.cpu().item()] += 1
    
    # normalize
    cm = cm / cm.sum(1).reshape(-1, 1)
    print(cm)

    classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    import itertools
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig('confusion_matrix.png', dpi=400)


def saliency_map(device, model, dataset, num_images):
    model.eval()
    targets = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}
    for batch_idx, (x, y) in enumerate(dataset):
        if batch_idx == num_images:
            break
        print(f'Processing image {batch_idx}...')
        # fetch an image
        x, y = x.to(device), y.to(device)
        x.requires_grad_()        

        # get gradient for input
        output = model(x)
        output_idx = output.argmax()
        output_max = output[0, output_idx]
        output_max.backward()

        # get saliency map
        saliency, _ = torch.max(x.grad.data, dim=1)
        saliency = torch.where(saliency >= 0.0, saliency, torch.Tensor([0]).to(torch.float32).to(device))
        saliency = saliency.reshape(48, 48)

        # get orig image
        image = x.cpu().detach().numpy().reshape(3, 48, 48)[0]
        image = (image * 0.229 + 0.485) * 255

        plt.figure()
        plt.subplot(121)
        plt.imshow(image, cmap='gray')
        plt.title(f'Label: {targets[y.cpu().item()]}')
        plt.subplot(122)
        plt.imshow(saliency.cpu(), cmap='hot')
        plt.title(f'Predicted: {targets[output_idx.cpu().item()]}')
        plt.tight_layout()
        plt.savefig(f'analyses/saliency/saliency_{batch_idx}_l{y.cpu().item()}_o{output_idx}.png', dpi=400)
        plt.close()


def lime(device, model, dataset, emotion, pred_correct=True, num_features=1, id=1):
    model.eval()
    targets = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}

    tgt_id = id
    for batch_idx, (x, y) in enumerate(dataset):
        x, y = x.to(device), y.to(device)
        out = model(x)
        y_pred = torch.argmax(F.softmax(out, dim=1), dim=1)
        assert y_pred.shape == y.shape
        if y.cpu().item() == emotion and (y_pred.item() == y.item()) == pred_correct:
            if id > 1:
                id -= 1
            else:
                break
    
    
    # get original image
    image = x.cpu().detach().numpy().reshape(3, 48, 48)[0]
    image = (image * 0.229 + 0.485) * 255

    #lime
    def batch_predict(images):
        batch = torch.stack(tuple(torch.Tensor(i).reshape(3, 48, 48) for i in images), dim=0)
        mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1).repeat(1, 48, 48)
        std = torch.Tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1).repeat(1, 48, 48)
        batch = (batch - mean) / std

        logits = model(batch.to(device))
        probs = F.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(image.astype('double'), 
                                            batch_predict, # classification function
                                            top_labels=5, 
                                            hide_color=0, 
                                            num_samples=1000)
    image_out, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=num_features, hide_rest=False)
    print(mask.shape, mask.sum())
    img_boundry1 = mark_boundaries(image_out / 255.0, mask)
    plt.subplot(121)
    plt.title(f'Label: {targets[y.cpu().item()]}')
    plt.imshow(image, cmap='gray')
    plt.subplot(122)
    plt.title(f'Predicted: {targets[y_pred.cpu().item()]}')
    plt.imshow(img_boundry1)
    plt.savefig(f'analyses/lime/{targets[emotion]}_{pred_correct}_{tgt_id}.png', dpi=400)


if __name__ == '__main__':
    main()

