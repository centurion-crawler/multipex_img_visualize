**Train**: python train.py -net sam -mod sam_adpt -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -image_size 256
**Inference**: python eval_vis.py -net sam -mod sam_adpt -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -weights <MED-SAM-IMC-weight.pth> -image_size 256 
