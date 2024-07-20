import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm

from utils.eval import eval_net
from lib.DS_TransUNet import UNet

from torch.utils.data import DataLoader, random_split
from utils.dataloader import get_loader,test_dataset

# train_img_dir = 'data/Kvasir_SEG/train/image/'
# train_mask_dir = 'data/Kvasir_SEG/train/mask/'
# val_img_dir = 'data/Kvasir_SEG/val/images/'
# val_mask_dir = 'data/Kvasir_SEG/val/masks/'
dir_checkpoint = 'checkpoints/0615-fold0/'


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def cal(loader):
    tot = 0
    for batch in tqdm(loader):
        imgs = batch['img_t'] 
        true_masks = batch['target_t']
        img_name = batch['img_name']
        # print(img_name)
        hs = batch['hs']
        ws = batch['ws']
        # print(imgs.shape)
        tot += imgs.shape[0]
    return tot

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=15, stride=1, padding=7) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay

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

def train_net(net,
              device,
              epochs=200,
              batch_size=16,
              lr=0.01,
              save_cp=True,
              n_class=1,
              img_size=512):


    train_loader = get_loader(image_size=img_size, mean_std_path='/home/zzf/dataset/IMC_Cell/Dice-XMBD/datas/melanoma/ch_dict.pkl', folds_path='/mnt/mydisk/zzf/code/Medical-SAM-Adapter-main/data/Melanoma_instance/train_val_test.pkl', fold_i=0,mode='Training',batchsize=batch_size)
    val_loader = get_loader(image_size=img_size, mean_std_path='/home/zzf/dataset/IMC_Cell/Dice-XMBD/datas/melanoma/ch_dict.pkl', folds_path='/mnt/mydisk/zzf/code/Medical-SAM-Adapter-main/data/Melanoma_instance/train_val_test.pkl', fold_i=0,mode='Validing',batchsize=batch_size)

    n_train = cal(train_loader)
    n_val = cal(val_loader)
    logger = get_logger('melanoma_0615.log')
    # n_train = len(train_loader)
    # n_val = len(val_loader)

    logger.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Vailding size:   {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images size:  {img_size}
    ''')

    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs//5, lr/10)
    
    if n_class > 1:
        criterion = nn.CrossEntropyLoss(weight = torch.Tensor([0.025]+[1]*20).to(device))
    else:
        criterion = nn.BCEWithLogitsLoss()


    best_dice = 0
    size_rates = [384]
    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        b_cp = False
        Batch = len(train_loader)
        with tqdm(total=n_train*len(size_rates), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                for rate in size_rates:
                    imgs = batch['img_t'] 
                    true_masks = batch['target_t']
                    img_name = batch['img_name']
                    # print(img_name)
                    hs = batch['hs']
                    ws = batch['ws']
                    trainsize = rate
                    # if rate != 512:
                    # print(imgs.shape,true_masks.shape)
                    
                    # imgs = F.upsample(imgs, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    # true_masks_cell = true_masks.clone()
                    # true_masks_cell[true_masks_cell>0]=1

                    true_masks = make_one_hot(true_masks,21).float()
                    # true_masks_cell = make_one_hot(true_masks_cell,2).float()

                    # true_masks = F.upsample(true_masks, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    # true_masks_cell = F.upsample(true_masks_cell, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

                    imgs = imgs.to(device=device, dtype=torch.float32)
                    mask_type = torch.float32 if n_class == 1 else torch.long
                    true_masks = true_masks.to(device=device, dtype=mask_type)
                    # true_masks_cell = true_masks_cell.to(device=device, dtype=mask_type)

                    # o_cls, o_cell, loss1, loss2_cls, loss2_cell
                    # masks_pred_cls, masks_pred_cell, l2_cls, l2_cell, l3_cls, l3_cell = net(imgs)
                    masks_pred_cls, l2_cls, l3_cls= net(imgs)

                    loss1_cls = structure_loss(masks_pred_cls, true_masks.float())
                    loss2_cls = structure_loss(l2_cls, true_masks.float())
                    loss3_cls = structure_loss(l3_cls, true_masks.float())

                    # loss1_cell = structure_loss(masks_pred_cell, true_masks_cell.float())
                    # loss2_cell = structure_loss(l2_cell, true_masks_cell.float())
                    # loss3_cell = structure_loss(l3_cell, true_masks_cell.float())

                    num_masks = torch.argmax(true_masks,dim=1)
                    # loss = 0.6*(loss1_cls+loss1_cell) + 0.2*(loss2_cls+loss2_cell) + 0.2*(loss3_cls+loss3_cell)  #+ criterion(masks_pred[:,1:],num_masks)
                    loss = 0.6*(loss1_cls) + 0.2*(loss2_cls) + 0.2*(loss3_cls)  + criterion(masks_pred_cls,num_masks)
                    # loss = criterion(masks_pred_cls,num_masks)
                    epoch_loss += loss.item()

                    pbar.set_postfix(**{'loss (batch)': loss.item()})

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_value_(net.parameters(), 0.1)
                    optimizer.step()

                    pbar.update(imgs.shape[0])

        scheduler.step()
        val_dice = eval_net(net, val_loader, device, n_class=21)
        if val_dice > best_dice:
           best_dice = val_dice
           b_cp = True
        epoch_loss = epoch_loss / Batch
        logger.info('epoch: {} train_loss: {:.3f} epoch_dice: {:.3f}, best_dice: {:.3f}'.format(epoch + 1, epoch_loss, val_dice* 100, best_dice * 100))

        if save_cp and b_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + 'epoch:{}_dice:{:.3f}.pth'.format(epoch + 1, val_dice*100))
            logging.info(f'Checkpoint {epoch + 1} saved !')



def get_args():
    parser = argparse.ArgumentParser(description='Train the model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=200,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=8,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=None,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--img_size', dest='size', type=int, default=224,
                        help='The size of the images')
    parser.add_argument('--optimizer', type=str,
                        default='Adam', help='choosing optimizer Adam or SGD')
    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = UNet(128, 21)
    net = nn.DataParallel(net, device_ids=[2])
    net = net.to(device)

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
    logging.info(f'Model loaded from {args.load}')

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_size=args.size,
                  n_class=21
                  )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
