import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm


def dice_coeff(pred, gt, smooth=1, activation='sigmoid'):
    r""" computational formula：
        dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d activation function operation")

    #pred = activation_fn(pred)

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    loss = 2 * (intersection + smooth) / (unionset + smooth)
    return loss.sum(), N

def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.

    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    masks=[]
    for i in range(0,num_classes):
        masks.append((input==i).long())
    
    result = torch.stack(masks,dim=1)

    return result

def eval_net(net, loader, device, n_class=1):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if n_class == 1 else torch.long
    tot = 0
    n_val = len(loader) 
    N = 0
    trainsize = 384
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs = batch['img_t'] 
            true_masks = batch['target_t']
            img_name = batch['img_name']
            hs = batch['hs']
            ws = batch['ws']
            # imgs = F.upsample(imgs, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            true_masks = make_one_hot(true_masks,21).float()
            # true_masks = F.upsample(true_masks, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)
            
            mask_pred, _, _ = net(imgs)

            # if n_class > 1:
            #     tot += F.cross_entropy(mask_pred, true_masks).item()
            # else:
            pred = torch.sigmoid(mask_pred)
            pred = (pred > 0.5).float()
            l, n = dice_coeff(pred, true_masks)
            tot += l
            N += n
            pbar.update()

    return tot / N

# def visualize_net()