# Copyright by ROS-SAM team
# All rights reserved.


import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import cv2
import random
from typing import  List, Tuple
from segment_anything_training import sam_model_registry
from segment_anything_training.modeling import TwoWayTransformer, MaskDecoder
from torch.utils.data import DataLoader
from utils.loss_mask import loss_masks
import utils.misc as misc
import skimage.io as skio
import numpy as np
from mmdet.rossamdataset import ROSSAM_Dataset
from safetensors import safe_open
from tqdm import tqdm
from safetensors.torch import save_file
import warnings
warnings.filterwarnings('ignore')
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs,InitProcessGroupKwargs
from accelerate.utils import set_seed

class LoRA_qkv(nn.Module):
    """
    LoRA adaption for attention modules. Only for queries and values

    Arguments:
        qkv: Original block of attention
        linear_a_q: linear block for q
        linear_b_q: linear block for q
        linear_a_v: linear block for v
        linear_b_v: linear block for v

    Return:
        qkv(nn.Module): qkv block with all linear blocks added (equivalent to adding the matrix B*A)
    """

    def __init__(
            self,
            qkv,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        
        super(LoRA_qkv, self).__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.d_model = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)
        q_ba = self.linear_b_q(self.linear_a_q(x))
        v_ba = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, :self.d_model] += q_ba #q part
        qkv[:, :, :, -self.d_model:] += v_ba #v part

        return qkv


class LoRA_sam(nn.Module):
    """
    Class that takes the image encoder of SAM and add the lora weights to the attentions blocks

    Arguments:
        sam_model: Sam class of the segment anything model
        rank: Rank of the matrix for LoRA
        lora_layer: List of weights exisitng for LoRA
    
    Return:
        None

    """

    def __init__(self, sam_model, rank, lora_layer=None):
        super(LoRA_sam, self).__init__()
        self.rank = rank
        assert rank > 0
        # base_vit_dim = sam_model.image_encoder.patch_embed.proj.out_channels

        if lora_layer:
            self.lora_layer = lora_layer
        else:
            # In each block, you have an attention block => total blocks -> nb lora layers
            self.lora_layer = list(range(len(sam_model.image_encoder.blocks)))
        
        self.A_weights = []
        self.B_weights = []

        # freeze parameters of the image encoder
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            # if only lora on few layers
            if t_layer_i not in self.lora_layer:
                continue

            w_qkv_linear = blk.attn.qkv
            self.d_model = w_qkv_linear.in_features

            w_a_linear_q = nn.Linear(self.d_model, self.rank, bias=False)
            w_b_linear_q = nn.Linear(self.rank, self.d_model, bias=False)
            w_a_linear_v = nn.Linear(self.d_model, self.rank, bias=False)
            w_b_linear_v = nn.Linear(self.rank, self.d_model, bias=False)
            

            self.A_weights.append(w_a_linear_q)
            self.B_weights.append(w_b_linear_q)
            self.A_weights.append(w_a_linear_v)
            self.B_weights.append(w_b_linear_v)

            blk.attn.qkv = LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v
            )

        self.reset_parameters()
        self.sam = sam_model
        self.lora_vit = sam_model.image_encoder


    def reset_parameters(self):
        """
        Initialize the LoRA A and B matrices like in the paper
        """
        # Initalisation like in the paper
        for w_A in self.A_weights:
            nn.init.kaiming_uniform_(w_A.weight, a=np.sqrt(5))
        for w_B in self.B_weights:
            nn.init.zeros_(w_B.weight)


    def save_lora_parameters(self, filename: str):
        """
        Save the LoRA wieghts applied to the attention model as safetensors.

        Arguments:
            filenmame: Name of the file that will be saved
        
        Return:
            None: Saves a safetensors file
        """
        num_layer = len(self.A_weights)
        # sufix 03:d -> allows to have a name 1 instead of 001
        a_tensors = {f"w_a_{i:03d}": self.A_weights[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.B_weights[i].weight for i in range(num_layer)}
        merged_dict = {**a_tensors, **b_tensors}
        save_file(merged_dict, filename)


    def load_lora_parameters(self, filename: str):
        """
        Load a safetensor file of LoRA weights for the attention modules

        Arguments:
            filename: Name of the file containing the saved weights
        
        Return:
            None: Loads the weights to the LoRA_sam class
        """
        with safe_open(filename, framework="pt") as f:
            for i, w_A_linear in enumerate(self.A_weights):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_A_linear.weight = nn.Parameter(saved_tensor)

            for i, w_B_linear in enumerate(self.B_weights):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_B_linear.weight = nn.Parameter(saved_tensor)
    def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
        for n, p in model.named_parameters():
            # print(n)
            if 'linear_' not in n:
                p.requires_grad = False
        if bias == 'none':
            return
        elif bias == 'all':
            for n, p in model.named_parameters():
                if 'bias' in n:
                    p.requires_grad = True
        elif bias == 'lora_only':
            for m in model.modules():
                if isinstance(m, LoRALayer) and \
                    hasattr(m, 'bias') and \
                    m.bias is not None:
                        m.bias.requires_grad = True
        else:
            raise NotImplementedError
    
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

class MaskDecoderHQ(MaskDecoder):
    def __init__(self, model_type):
        super().__init__(transformer_dim=256,
                        transformer=TwoWayTransformer(
                                depth=2,
                                embedding_dim=256,
                                mlp_dim=2048,
                                num_heads=8,
                            ),
                        num_multimask_outputs=3,
                        activation=nn.GELU,
                        iou_head_depth= 3,
                        iou_head_hidden_dim= 256,)
        assert model_type in ["vit_b","vit_l","vit_h"]
        
        checkpoint_dict = {"vit_b":"pretrained_checkpoint/sam_vit_b_maskdecoder.pth",
                           "vit_l":"pretrained_checkpoint/sam_vit_l_maskdecoder.pth",
                           'vit_h':"pretrained_checkpoint/sam_vit_h_maskdecoder.pth"}
        checkpoint_path = checkpoint_dict[model_type]
        self.load_state_dict(torch.load(checkpoint_path))
        print("HQ Decoder init from SAM MaskDecoder")
        for n,p in self.named_parameters():
            p.requires_grad = False

        transformer_dim=256
        vit_dim_dict = {"vit_b":768,"vit_l":1024,"vit_h":1280}
        vit_dim = vit_dim_dict[model_type]

        self.hf_token = nn.Embedding(1, transformer_dim)
        self.hf_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        self.num_mask_tokens = self.num_mask_tokens + 1

        self.compress_vit_feat = nn.Sequential(
                                        nn.ConvTranspose2d(vit_dim, transformer_dim, kernel_size=2, stride=2),
                                        LayerNorm2d(transformer_dim),
                                        nn.GELU(), 
                                        nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2))

        self.embedding_encoder = nn.Sequential(
                                        nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
                                        LayerNorm2d(transformer_dim // 4),
                                        nn.GELU(),
                                        nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
                                    )

        self.embedding_maskfeature = nn.Sequential(
                                        nn.Conv2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1), 
                                        LayerNorm2d(transformer_dim // 4),
                                        nn.GELU(),
                                        nn.Conv2d(transformer_dim // 4, transformer_dim // 8, 3, 1, 1))

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        hq_token_only: bool,
        interm_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the ViT image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted hq masks
        """

        vit_features = interm_embeddings[0].permute(0, 3, 1, 2) # early-layer ViT feature, after 1st global attention block in ViT        
        hq_features = self.embedding_encoder(image_embeddings) + self.compress_vit_feat(vit_features)

        batch_len = len(image_embeddings)
        masks = []
        iou_preds = []
        for i_batch in range(batch_len):
            mask, iou_pred = self.predict_masks(
                image_embeddings=image_embeddings[i_batch].unsqueeze(0),
                image_pe=image_pe[i_batch],
                sparse_prompt_embeddings=sparse_prompt_embeddings[i_batch],
                dense_prompt_embeddings=dense_prompt_embeddings[i_batch],
                hq_feature = hq_features[i_batch].unsqueeze(0)
            )
            masks.append(mask)
            iou_preds.append(iou_pred)
        masks = torch.cat(masks,0)
        iou_preds = torch.cat(iou_preds,0)

        # Select the correct mask or masks for output
        if multimask_output:
            # mask with highest score
            mask_slice = slice(1,self.num_mask_tokens-1)
            iou_preds = iou_preds[:, mask_slice]
            iou_preds, max_iou_idx = torch.max(iou_preds,dim=1)
            iou_preds = iou_preds.unsqueeze(1)
            masks_multi = masks[:, mask_slice, :, :]
            masks_sam = masks_multi[torch.arange(masks_multi.size(0)),max_iou_idx].unsqueeze(1)
        else:
            # singale mask output, default
            mask_slice = slice(0, 1)
            masks_sam = masks[:,mask_slice]

        masks_hq = masks[:,slice(self.num_mask_tokens-1, self.num_mask_tokens), :, :]
        
        if hq_token_only:
            return masks_hq
        else:
            return masks_sam, masks_hq

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        hq_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""

        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.hf_token.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0) 
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)

        upscaled_embedding_sam = self.output_upscaling(src)
        upscaled_embedding_ours = self.embedding_maskfeature(upscaled_embedding_sam) + hq_feature
        
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            if i < 4:
                hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
            else:
                hyper_in_list.append(self.hf_mlp(mask_tokens_out[:, i, :]))

        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding_sam.shape

        masks_sam = (hyper_in[:,:4] @ upscaled_embedding_sam.view(b, c, h * w)).view(b, -1, h, w)
        masks_ours = (hyper_in[:,4:] @ upscaled_embedding_ours.view(b, c, h * w)).view(b, -1, h, w)
        masks = torch.cat([masks_sam,masks_ours],dim=1)
        
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


def get_args_parser():
    parser = argparse.ArgumentParser('ROS-SAM', add_help=False)

    parser.add_argument("--output", type=str, default='result/temptest', 
                        help="Path to the directory where masks and checkpoints will be output")
    parser.add_argument("--data-dir", type=str, default='/opt/data/workspace/shanzhe/SATMTB-HQ', 
                        help="The path of dataset.")    
    parser.add_argument("--model-type", type=str, default="vit_l", 
                        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--checkpoint", type=str,default='pretrained_checkpoint/sam_vit_l_0b3195.pth', 
                        help="The path to the SAM checkpoint to use for mask generation.")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="The device to run generation on.")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--lr_drop_epoch', default=4, type=int)
    parser.add_argument('--max_epoch_num', default=48, type=int)
    parser.add_argument('--input_size', default=512, type=int)
    parser.add_argument('--batch_size_train', default=1, type=int)
    parser.add_argument('--batch_size_valid', default=1, type=int)
    parser.add_argument('--model_save_fre', default=3, type=int)

    parser.add_argument('--eval', default=True, action='store_true')
    parser.add_argument('--visualize', default=True, action='store_true')
    parser.add_argument("--restore-model", type=str,default=None,
                        help="The path to the hq_decoder training checkpoint for evaluation")
    return parser.parse_args()


def main(net, args):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs],project_dir=args.output)
    set_seed(args.seed)

    ### --- Step 1: Loading data---

    train_datasets = ROSSAM_Dataset(args.data_dir,'train')
    train_dataloaders = DataLoader(train_datasets,batch_size=1,shuffle=True,num_workers=8,drop_last=True)
    if accelerator.is_main_process:
        print("--- create training dataloader ---")
        print(len(train_dataloaders), " train dataloaders created")

    valid_datasets = ROSSAM_Dataset(args.data_dir,'val')
    valid_dataloaders = DataLoader(valid_datasets,batch_size=1,shuffle=False,num_workers=8,drop_last=True )
    if accelerator.is_main_process:
        print("--- create valid dataloader ---")
        print(len(valid_dataloaders), " valid dataloaders created")
    
    ### --- Step 2: Building the model---

    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    LoRA_sam(sam,512)
    inputsam = sam.to(device=args.device)
    net = net.to(device=args.device)
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    for name, param in sam.named_parameters():
        if 'image_encoder.blocks.23' in name:
            param.requires_grad = True
        elif 'mask_decoder'  in name:
            param.requires_grad = True                 

    optimizer_sam = optim.Adam(sam.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop_epoch)
    lr_scheduler.last_epoch = args.start_epoch
    net, inputsam, optimizer,optimizer_sam, train_dataloaders, valid_dataloaders, lr_scheduler = accelerator.prepare(net,inputsam, optimizer, optimizer_sam, train_dataloaders, valid_dataloaders, lr_scheduler)
    ### --- Step 3: Train or Evaluate ---
    train(args, inputsam, net, optimizer,optimizer_sam, train_dataloaders, valid_dataloaders, lr_scheduler, accelerator)

def train(args, sam , net, optimizer, optimizer_sam, train_dataloaders, valid_dataloaders, lr_scheduler, accelerator):
    if accelerator.is_main_process:
        os.makedirs(args.output, exist_ok=True)

    epoch_start = args.start_epoch
    epoch_num = args.max_epoch_num
    net.train()
    sam =sam.train() 
    # for name, param in sam.named_parameters():
    #     print(f"Parameter: {name}, Gradient: {param.requires_grad}")
    # for name, param in net.named_parameters():
    #     print(f"Parameter: {name}, Gradient: {param.requires_grad}")
    for epoch in range(epoch_start,epoch_num): 
        allloss,allloss_mask,allloss_dice = 0,0,0
        if accelerator.is_main_process:
            print("epoch:   ",epoch, "  learning rate:  ", optimizer.param_groups[0]["lr"])
        for iteridx,data in enumerate(tqdm(train_dataloaders,disable=not accelerator.is_local_main_process,miniters=20 )):
            skipsignal = 0
            inputs  = data['img']
            boxes, onehotmask = data['boxes'], data['onehotmask']    
            if len(boxes[0]) == 0:  
                boxes = torch.zeros((1, 1, 4),dtype=torch.float32).to(device=inputs.device)
                onehotmask = onehotmask[:,0:1,:]
                if onehotmask.size()[1] != 1:
                    onehotmask = torch.zeros((1, 1, 1024, 1024),dtype=torch.float32).to(device=inputs.device)
                skipsignal = 1
            elif 'car' in data['filename'][0]:
                boxes = torch.zeros((1, 1, 4),dtype=torch.float32).to(device=inputs.device)
                onehotmask = onehotmask[:,0:1,:]
                if onehotmask.size()[1] != 1:
                    onehotmask = torch.zeros((1, 1, 1024, 1024),dtype=torch.float32).to(device=inputs.device)
                skipsignal =1 
            elif boxes.size()[1] != onehotmask.size()[1]:
                boxes = torch.zeros((1, 1, 4),dtype=torch.float32).to(device=inputs.device)
                onehotmask = onehotmask[:,0:1,:]
                if onehotmask.size()[1] != 1:
                    onehotmask = torch.zeros((1, 1, 1024, 1024),dtype=torch.float32).to(device=inputs.device)
                skipsignal = 1
  
            imgs = inputs.permute(0, 2, 3, 1).cpu().numpy()       
            input_keys = ['box']
            labels_box =boxes
            batched_input = []

            for b_i in range(len(imgs)):
                dict_input = dict()
                input_image = torch.as_tensor(imgs[b_i].astype(dtype=np.uint8), device=sam.device).permute(2, 0, 1).contiguous()
                dict_input['image'] = input_image 
                input_type = random.choice(input_keys)
                if input_type == 'box':
                    dict_input['boxes'] = torch.squeeze(labels_box[b_i:b_i+1],dim=0)
                else:
                    raise NotImplementedError
                dict_input['original_size'] = imgs[b_i].shape[:2]
                batched_input.append(dict_input)
                sam_input = np.squeeze(imgs.astype(np.uint8),axis=0)
                batched_output, interm_embeddings = sam(batched_input, multimask_output=False)                        
                batch_len = len(batched_output)
                encoder_embedding = torch.cat([batched_output[i_l]['encoder_embedding'] for i_l in range(batch_len)], dim=0)
                image_pe = [batched_output[i_l]['image_pe'] for i_l in range(batch_len)]
                sparse_embeddings = [batched_output[i_l]['sparse_embeddings'] for i_l in range(batch_len)]
                dense_embeddings = [batched_output[i_l]['dense_embeddings'] for i_l in range(batch_len)]
                mask = batched_output[0]['masks']

            masks_hq = net(
                image_embeddings=encoder_embedding,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                hq_token_only=True,
                interm_embeddings=interm_embeddings,
            )
            onehotmask = onehotmask.permute(1,0,2,3)
            loss_mask, loss_dice = loss_masks(masks_hq, onehotmask/255.0, len(masks_hq))
            loss = loss_mask + loss_dice
            if skipsignal == 1:
                loss = loss*0

            # print('-------------------------------------------------------------------')
            # print((sam.state_dict()['image_encoder.blocks.11.attn.qkv.linear_a_q.weight']))
            # print('-------------------------------------------------------------------')
            # print((sam.state_dict()['image_encoder.blocks.11.attn.qkv.qkv.weight']))
            # print('-------------------------------------------------------------------')                    
            optimizer.zero_grad()
            optimizer_sam.zero_grad()
            accelerator.backward(loss)

            if iteridx%2 ==0:
                optimizer_sam.step()
            else :
                optimizer.step()
            allloss += float(loss.detach().cpu().numpy())
            allloss_mask += float(loss_mask.detach().cpu().numpy())
            allloss_dice += float(loss_dice.detach().cpu().numpy())
            if accelerator.is_main_process and iteridx % 100 == 0:
                loadimgs= np.array(sam_input.copy().astype(np.uint8))
                loadgt = np.array(onehotmask.cpu())
                loadgt = np.max(loadgt, axis=0)
                loadgt = np.where(loadgt > 0, 255, 0)
                loadgt = np.squeeze(loadgt)
                loadgt = np.array(loadgt,dtype=np.uint8)                                
                for box in labels_box[0]:
                    x_min, y_min, x_max, y_max = int(box[0]),int(box[1]),int(box[2]),int(box[3])
                    cv2.rectangle(loadgt, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (124), thickness=1)  
                samoutput = mask.cpu().numpy()
                samoutput = samoutput.astype(np.uint8) 
                if len(samoutput.shape) == 4:
                    samoutput = np.squeeze(samoutput,axis=1)
                samoutput = np.max(samoutput, axis=0)
                samoutput = np.where(samoutput > 0, 255, 0)
                samoutput = np.squeeze(samoutput)
                samoutput = np.array(samoutput,dtype=np.uint8)
                for box in labels_box[0]:
                    x_min, y_min, x_max, y_max = int(box[0]),int(box[1]),int(box[2]),int(box[3])
                    cv2.rectangle(samoutput, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (124), thickness=1)                            
                masks_hq_vis = (F.interpolate(masks_hq.detach(), (loadimgs.shape[0], loadimgs.shape[1]), mode="bilinear", align_corners=False) > 0).cpu().numpy()    
                masks_hq_vis = np.squeeze(masks_hq_vis,axis=1)
                masks_hq_vis = np.max(masks_hq_vis, axis=0)
                masks_hq_vis = np.where(masks_hq_vis > 0, 255, 0)
                masks_hq_vis = np.squeeze(masks_hq_vis)
                masks_hq_vis = np.array(masks_hq_vis,dtype=np.uint8)
                for box in labels_box[0]:
                    x_min, y_min, x_max, y_max = int(box[0]),int(box[1]),int(box[2]),int(box[3])
                    cv2.rectangle(masks_hq_vis, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (124), thickness=1)                        
                os.makedirs(os.path.join(args.output,f'trainvis/sam/{epoch}'),exist_ok=True)
                os.makedirs(os.path.join(args.output,f'trainvis/hqsam/{epoch}'),exist_ok=True)
                os.makedirs(os.path.join(args.output,f'trainvis/gts/{epoch}'),exist_ok=True)
                skio.imsave(os.path.join(os.path.join(args.output,f'trainvis/gts/{epoch}'),str(str(iteridx)+'_'+data['filename'][0])),loadgt)
                skio.imsave(os.path.join(os.path.join(args.output,f'trainvis/sam/{epoch}'),str(str(iteridx)+'_'+data['filename'][0])),samoutput)
                skio.imsave(os.path.join(os.path.join(args.output,f'trainvis/hqsam/{epoch}'),str(str(iteridx)+'_'+data['filename'][0])),masks_hq_vis)

        avg_loss = allloss/(iteridx+1)
        avg_loss_mask = allloss_mask/(iteridx+1)
        avg_loss_dice = allloss_dice/(iteridx+1)
        lr_scheduler.step()        
        if accelerator.is_main_process:
            with open(os.path.join(args.output,'train_stats.txt'), 'a') as file:
                file.write(f'Epoch {epoch} stats:\n')  
                file.write(f'training_loss: {avg_loss:.4f}  loss_dice: {avg_loss_mask:.4f}  loss_mask: {avg_loss_dice:.4f}\n') 
        accelerator.save_state(os.path.join(args.output,f'checkpoint_{epoch}'))  
    print("Training Reaches The Maximum Epoch Number")
    

if __name__ == "__main__":
    args = get_args_parser()
    net = MaskDecoderHQ(args.model_type) 
    main(net, args)