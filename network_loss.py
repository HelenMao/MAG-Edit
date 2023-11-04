from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm import tqdm
import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn.functional as nnf
import numpy as np
import abc
import ptp_utils1
import seq_aligner
import shutil
from torch.optim.adam import Adam
from PIL import Image
from torchvision import utils 
import os
from math import sqrt
import cv2
from gaussian_smoothing import GaussianSmoothing
torch.autograd.set_detect_anomaly(True)
import torchvision.transforms as transforms
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
MY_TOKEN = ''
LOW_RESOURCE = False 
NUM_DDIM_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
ldm_stable = StableDiffusionPipeline.from_pretrained("/home/user/stable-diffusion-v1-4/", scheduler=scheduler).to(device)
try:
    ldm_stable.disable_xformers_memory_efficient_attention()
except AttributeError:
    print("Attribute disable_xformers_memory_efficient_attention() is missing")
tokenizer = ldm_stable.tokenizer

print(ldm_stable.scheduler.config.steps_offset)




class LocalBlend:
    
    def get_mask(self, maps, alpha, use_pool,x_t=None,t=0):
        k = 1
        maps = (maps * alpha).sum(-1).mean(1)
        if use_pool:
            maps = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(maps, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        if t%10 ==0 and self.name:
            utils.save_image(mask, "{}/1-{}.png".format(self.name,t//10), nrow=2)        
        mask = mask.gt(self.th[1-int(use_pool)])
        #mask = mask[:1] + mask
        return mask
    
    def __call__(self, x_t, attention_store,t):
        self.counter += 1
        if self.counter > self.start_blend:
           
            maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
            maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
            maps = torch.cat(maps, dim=1)
            mask = self.get_mask(maps, self.alpha_layers, True,x_t,self.counter)
            if self.substruct_layers is not None:
                maps_sub = ~self.get_mask(maps, self.substruct_layers, False)
                mask = mask * maps_sub
            mask = mask.float()
            x_t = x_t[:1] + mask * (x_t - x_t[:1])
            if self.counter%10 ==0 and self.name:
                utils.save_image(mask, "{}/2-{}.png".format(self.name,self.counter//10), nrow=2) 

        return x_t
       
    def __init__(self, prompts: List[str], words: [List[List[str]]], substruct_words=None, start_blend=0.2, th=(.3, .3),name=None):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils1.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1

        if substruct_words is not None:
            substruct_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
            for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
                if type(words_) is str:
                    words_ = [words_]
                for word in words_:
                    ind = ptp_utils1.get_word_inds(prompt, word, tokenizer)
                    substruct_layers[i, :, :, :, :, ind] = 1
            self.substruct_layers = substruct_layers.to(device)
        else:
            self.substruct_layers = None
        self.alpha_layers = alpha_layers.to(device)
        self.start_blend = int(start_blend * NUM_DDIM_STEPS)
        self.counter = 0 
        self.th=th
        if name:
            self.name = os.path.join(name,"mask")
            os.makedirs(self.name,exist_ok=True)
        
        
class EmptyControl:
    
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn

    
class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t,t=0):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
    
      # optimizing
       if self.optimizing:
          # reconstruction branch
          if self.total % 2 == 0:
             self.step_atten={"down_cross": [],  "up_cross": [],  "mid_cross": []}
             
           
             
             h = attn.shape[0]
             attn = self.save_recon_attn(attn,is_cross,place_in_unet)
             self.cur_recon_layer += 1
             
             if self.cur_recon_layer == self.num_att_layers + self.num_uncond_att_layers:
               self.total+=1
               self.cur_recon_layer = 0
          
          # edit branch   
          else:
             
              #if self.cur_optimize_attn_layer == 0:
                  #self.step_atten={"down_cross": [],  "up_cross": [],  "mid_cross": []}
              
              attn = self.forward(attn, is_cross, place_in_unet)
              #attn = renormalize2(attn.detach(),self.prompt)
              #save atten for loss computation
              #self.save_optimize_attn(attn,is_cross,place_in_unet)
              
              self.cur_optimize_attn_layer += 1
              if self.cur_optimize_attn_layer == self.num_att_layers:
                  self.cur_optimize_attn_layer = 0
                  self.between_steps()    
                  self.total+=1      
          
          
          #not optimizing
       else:  
          # reconstruction branch
          if self.total % 2 == 0:
               
             h = attn.shape[0]
             attn[h // 2:] = self.save_recon_attn(attn[h // 2:],is_cross,place_in_unet)
             self.cur_recon_layer += 1
             
             if self.cur_recon_layer == self.num_att_layers + self.num_uncond_att_layers:
               self.total+=1
               self.cur_recon_layer = 0
              
          #edit branch
          else:          
             
             h = attn.shape[0]
             
             #replace attention map
             attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
             self.cur_att_layer += 1
                
             if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
                self.cur_att_layer = 0
                self.cur_step += 1
                self.between_steps()
                self.total  +=1       
                  
                  
       return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0
        self.cur_optimize_attn_layer = 0
        
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.cur_optimize_attn_layer = 0
        self.total=0
        self.cur_recon_layer = 0
        
class SpatialReplace(EmptyControl):
    
    def step_callback(self, x_t):
        if self.cur_step < self.stop_inject:
            b = x_t.shape[0]
            x_t = x_t[:1].expand(b, *x_t.shape[1:])
        return x_t

    def __init__(self, stop_inject: float):
        super(SpatialReplace, self).__init__()
        self.stop_inject = int((1 - stop_inject) * NUM_DDIM_STEPS)
        

class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}
    def forward(self,attn, is_cross, place_in_unet):
        return attn

    @staticmethod
    def get_empty_count():
        return {"down_cross": 0, "mid_cross": 0, "up_cross": 0,
                "down_self": 0,  "mid_self": 0,  "up_self": 0}
    
    def save_recon_attn(self, attn, is_cross: bool, place_in_unet: str):
    
      key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
      '''if 'self' in key and attn.shape[1] <= 32 ** 2 and self.cur_step<30:  # avoid memory overhead
          self.step_store[key].append(attn)
      elif 'self' in key and self.cur_step>=30:
          self.step_store[key].append(attn) 
      elif 'cross' in key:
          self.step_store[key].append(attn)'''
      self.step_store[key].append(attn)
      return attn

  
 
    def save_v(self,v,place_in_unet):
      
        key = f"{place_in_unet}_{'self'}"
        #print(self.cur_step,key,v.size())
        self.v_store[key].append(v)   
             
        '''if v.size(1)<=32 **2:
          key = f"{place_in_unet}_{'self'}"
          self.v_store[key].append(v) ''' 
        
    def save_optimize_attn(self, attn, is_cross,place_in_unet):
       
        if is_cross:
            key = f"{place_in_unet}_{'cross'}"
            self.step_atten[key].append(attn)
    

            

    def between_steps(self):
        if not self.optimizing:
            if len(self.attention_store) == 0:
                self.attention_store = self.step_store
            else:
                for key in self.attention_store:
                    for i in range(len(self.attention_store[key])):
                        if 'self' in key:
                            self.attention_store[key][i] = self.attention_store[key][i].cpu()+self.step_store[key][i].cpu()
                        else:
                            self.attention_store[key][i] = self.attention_store[key][i].cpu()+ self.step_atten[key][i].cpu()
                        
        #if self.name and (self.cur_step+1) % 10==0:
         #self.show_cross()
       
                  
        self.step_store = self.get_empty_store()
        self.count_attn = self.get_empty_count()
        
    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in     self.attention_store}
        return average_attention
    
    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self,name=None):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.count_attn = self.get_empty_count()
        self.v_store = self.get_empty_store()
        self.attention_store = {}
        self.step_atten = {"down_cross": [],  "up_cross": [],  "mid_cross": []}
        self.name=name
        if self.name:
            os.makedirs(name,exist_ok=True)
        print(self.name)
        self.optimizing=False
        self.att_range=[] 
        self.m = 1


        
class AttentionControlEdit(AttentionStore, abc.ABC):
    
    def step_callback(self, x_t,t=0):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store,t)
        return x_t
        
    def replace_self_attention(self, attn ,place_in_unet):
    
        if attn.shape[2] <= 32 ** 2:  
            key = f"{place_in_unet}_{'self'}"
            attn_base = self.step_store[key][self.count_attn[key]]
            attn_base = attn_base.unsqueeze(0).expand(attn.shape[0], *attn_base.shape)
            return attn_base.detach()
        else:
           
            return attn
    
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            
      
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_repalce = attn
            
            if is_cross: 

                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_repalce, place_in_unet) * alpha_words + (1 - alpha_words) * attn_repalce
                self.save_optimize_attn(attn_repalce_new.reshape(self.batch_size * h, *attn.shape[2:]),is_cross,place_in_unet)
                
                #mask cross-attention
                w = int(sqrt(attn.shape[2]))
                #if not self.optimizing and self.adj_mask: 
                    #attn_repalce_new= self.mask_adj(attn_repalce_new.detach().reshape(1,h,w,w,attn.shape[-1])).reshape(1, *attn.shape[1:])
                
                attn = attn_repalce_new
            else:
                attn = self.replace_self_attention(attn, place_in_unet)
           
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
            
            
            self.count_attn[key] +=1        
        
        return attn
 

    def mask_adj(self,attn):
        
        if attn.size(2)==8:
            idx = 0
        elif attn.size(2)==16:
            idx = 1
        elif attn.size(2)==32:
            idx=2
        elif attn.size(2)==64:
            idx=3
        h =  attn.size(1) 
        sigma= 0.5
        sigma= 0.5
        kernel_size = 5
         
        smoothing = GaussianSmoothing(channels=h, kernel_size=kernel_size, sigma=sigma, dim=2).cuda()

                      
       
        for i in range(len(self.word[1])):
            if  self.word[1][i] not in self.prompt:
               continue             
            adj_alpha = ptp_utils1.get_word_inds(self.prompt, self.word[1][i], tokenizer)
            mask = self.adj_mask[self.word[0][i]][idx].unsqueeze(-1)
            attn_new = attn[:,:,:,:,adj_alpha] * mask
           
            #attn_new  = attn[:,:,:,:,adj_alpha] / 1000.0
            '''for i in range(len(adj_alpha)):
                temp = nnf.pad(attn_new[:,:,:,:,i], (2, 2, 2, 2),mode='constant', value=0)
                attn_new[:,:,:,:,i] = smoothing(temp)'''
                            
            attn[:,:,:,:,adj_alpha] = attn_new
            attn_new = attn[:,:,:,:,adj_alpha-1] * mask
            attn[:,:,:,:,adj_alpha-1] = attn_new
        #mask = (attn_new.squeeze(-1)[:,0,:,:]!=0).float()
        
        #mask padding
        #utils.save_image(mask, "{}/mask-{}.png".format(self.name,idx))   
        #last = ptp_utils1.get_word_inds(self.prompt, self.prompt.split()[-1], tokenizer)[-1]
        #print("last",last)
        #attn[:,:,:,:,last+1:] *= mask_padding.unsqueeze(-1)  
        
        '''if  idx == 1 and self.cur_step+1 in [3,5,8,15,10]:
            adj_alpha = ptp_utils1.get_word_inds(self.prompt, self.word[1][0], tokenizer)
            maps = attn[:,:,:,:,adj_alpha].sum(-1).mean(1).squeeze(0) 
            maps =  maps[self.adj_mask[self.word[0][0]][idx].squeeze()==1].cpu()
            self.att_range.append(maps)
            if self.cur_att_layer==19:
                np.save("{}/after{}-step{}.npy".format(self.name,self.word[1][0],self.cur_step+1),torch.stack(self.att_range).numpy()) 
                self.att_range=[]'''
            
        return attn            
    
 
            
    
    def mask_n(self,dir_name):
        #cross16 = torch.Tensor(cluster).to(device)
        adj_mask={}
        transform = transforms.ToTensor()
        for n  in self.word[0]:
        #for n  in ["mask"]:
            img = Image.open(os.path.join(dir_name,n)).convert('L')
            #n_alpha = self.get_n_alpha(n)
            maskk = transform(img)
            maskk = maskk.unsqueeze(0)            
            
            mask = nnf.interpolate(maskk, size=(8,8)).to(device)
            mask = mask!=0   
            adj_mask[n] = [mask]

            #utils.save_image(maskk, "{}/mask8-{}.png".format(self.name,n))    
       
            mask = nnf.interpolate(maskk, size=(16,16)).to(device)   
            mask = mask!=0   
            adj_mask[n].append(mask)
            utils.save_image(mask.float(), "{}/mask16-{}.png".format(self.name,n))   
            
            mask = nnf.interpolate(maskk, size=(32,32)).to(device)   
            mask = mask!=0   
            adj_mask[n].append(mask)
            #utils.save_image(mask, "{}/mask32-{}.png".format(self.name,n)) 
            
            mask = nnf.interpolate(maskk, size=(64,64))   
            
            mask = mask!=0 
              
            adj_mask[n].append(mask.to(device))
            #utils.save_image(mask.float() ,"{}/mask64-{}.png".format(self.name,n))
            
        return adj_mask



    
    def get_self_attn(self, attn,v,place_in_unet):
        w =  int(sqrt(v.shape[1]))
        if w==8:
          idx = 0
        elif w==16:
          idx = 1
        elif w==32:
          idx=2
        elif w==64:
          idx=3           
                   
        if  30 <= self.cur_step < 50 and w<= 32:
            key = f"{place_in_unet}_{'self'}"
            mask = torch.zeros(v.size(1)) 
            v_recon = self.step_store[key][self.count_attn[key]]
            #self.count_attn[key] +=1
            v_edit = v
            v_recon = self.v_store[key][self.count_attn[key]]
            
            
            for i in range(len(self.word[1])):
                adj_alpha = ptp_utils1.get_word_inds(self.prompt, self.word[1][i], tokenizer)
                mask_map = self.adj_mask[self.word[0][i]][idx].squeeze(0).reshape(1,w **2,-1)
                mask_tk_in = torch.unique(mask_map.nonzero(as_tuple=True)[1])     
                mask[mask_tk_in] =   1
            mask = mask.unsqueeze(-1).unsqueeze(0).to(device)           
          
            attn  = self.mask_self_attn_patches(attn,key)
            v_ = v_recon * (1-mask) + v_edit * mask
            self.count_attn[key] +=1
            
        else:
           v_ = v
           #attn = attn
           if 30 <= self.cur_step < 50:
             key = f"{place_in_unet}_{'self'}"
             self.count_attn[key] +=1
        return attn,v_


    def get_self_attn3(self, attn,v):
        w =  int(sqrt(v.shape[1]))
        if w==8:
          idx = 0
        elif w==16:
          idx = 1
        elif w==32:
          idx=2
        elif w==64:
          idx=3           
                   
        if  (25 <= self.cur_step <= 50 and w<=32):
            mask = torch.zeros(v.size(1)) 
            for i in range(len(self.word[1])):
                adj_alpha = ptp_utils1.get_word_inds(self.prompt, self.word[1][i], tokenizer)
                mask_map = self.adj_mask[self.word[0][i]][idx].squeeze(0).reshape(1,w **2,-1)
                mask_tk_in = torch.unique(mask_map.nonzero(as_tuple=True)[1])     
                mask[mask_tk_in] =   1
            mask = mask.unsqueeze(-1).unsqueeze(0).to(device)           
            
            attn  = self.mask_self_attn_patches(attn)
            v_ = v[:8] * (1-mask) + v[8:] * mask
            
        elif 30 <= self.cur_step <= 50 and w>32:
            '''mask = torch.zeros(v.size(1)) 
            for i in range(len(self.word[1])):
                adj_alpha = ptp_utils1.get_word_inds(self.prompt, self.word[1][i], tokenizer)
                mask_map = self.adj_mask[self.word[0][i]][idx].squeeze(0).reshape(1,w **2,-1)
                mask_tk_in = torch.unique(mask_map.nonzero(as_tuple=True)[1])     
                mask[mask_tk_in] =   1
            mask = mask.unsqueeze(-1).unsqueeze(0).to(device) '''          
            # first half of attn maps is for the uncoditioned image
            #attn  = self.mask_self_attn_patches(attn)
            #v_ = v[:8] * (1-mask) + v[8:] * mask
            attn = attn[:8]
            v_ = v[8:]
            #v_ = v[:8]
        else:
           v_ = v
           attn = attn
        return attn,v_





    
    def mask_self_attn_patches(self, self_attn,key):
        w = int(sqrt(self_attn.size(1)))
        #print("w",w)
        if w==8:
            idx = 0
        elif w==16:
            idx = 1
        elif w==32:
            idx=2
        elif w==64:
            idx=3
        h = 8
        mask = torch.zeros_like(self_attn[0]) 
        #print("mask",mask.size())
        attn_recon = self.step_store[key][self.count_attn[key]]
        attn_edit = self_attn
        for i in range(len(self.word[1])):
            adj_alpha = ptp_utils1.get_word_inds(self.prompt, self.word[1][i], tokenizer)
            mask_map = self.adj_mask[self.word[0][i]][idx].squeeze(0).reshape(1,w **2,-1)
            #print("mask",mask_map.size())
            mask_tk_in = torch.unique(mask_map.nonzero(as_tuple=True)[1])
            #print(mask_map.nonzero(as_tuple=True))
            #print(mask_map.nonzero(as_tuple=True).size())
            #utils.save_image(mask_map.nonzero(as_tuple=True)[0].float().squeeze(), "{}/selfmask-{}.png".format(self.name,w))
            mask[mask_tk_in, :] = 1
            mask[:, mask_tk_in] = 1
        attn = attn_edit *  mask + attn_recon* (1 - mask)
        
        return attn       


    def mask_self_attn_patches2(self, self_attn,key):
            w = int(sqrt(self_attn.size(1)))
            #print("w",w)
            if w==8:
                idx = 0
            elif w==16:
                idx = 1
            elif w==32:
                idx=2
            elif w==64:
                idx=3
            h = 8
            mask = torch.zeros_like(self_attn[0]) 
            #print("mask",mask.size())
            attn_recon = self.step_store[key][self.count_attn[key]]
            attn_edit = self_attn
            for i in range(len(self.word[1])):
                adj_alpha = ptp_utils1.get_word_inds(self.prompt, self.word[1][i], tokenizer)
                mask_map = self.adj_mask[self.word[0][i]][idx].squeeze(0).reshape(1,w **2,-1)
                #print("mask",mask_map.size())
                mask_tk_in = torch.unique(mask_map.nonzero(as_tuple=True)[1])
                #print(mask_map.nonzero(as_tuple=True))
                #print(mask_map.nonzero(as_tuple=True).size())
                #utils.save_image(mask_map.nonzero(as_tuple=True)[0].float().squeeze(), "{}/selfmask-{}.png".format(self.name,w))
                mask[mask_tk_in, :] = 1
                mask[:, mask_tk_in] = 1
            attn = attn_edit *  (1-mask) + attn_recon * mask
            
            return attn       


    
    
    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend]
                ,name=None,word=None,adj_mask=None,word_ng=None):
        
        super(AttentionControlEdit, self).__init__(name)
        self.batch_size = 1
        self.cross_replace_alpha = ptp_utils1.get_time_words_attention_alpha(prompts[:2], num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend
        self.word = word
        self.word_ng=word_ng
        self.prompt = prompts[1]
        print(prompts)
      
        self.ng_prompt = prompts[2] if word_ng is not None else None
    
        #self.adj_mask = self.mask_n(*adj_mask) if adj_mask else None
        self.adj_mask =  self.mask_n(adj_mask) if adj_mask is not None else None
        #print(self.num_self_replace)
        self.m = 0
        
        
class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, att_replace,place_in_unet):
        key = f"{place_in_unet}_{'cross'}"
        attn_base = self.step_store[key][self.count_attn[key]]
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
      
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None,name=None,word=None,adj_mask=None,word_ng=None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend,name=name,word=word,adj_mask=adj_mask,word_ng=word_ng)
        self.mapper = seq_aligner.get_replacement_mapper(prompts[:2], tokenizer).to(device)
        print(self.mapper)

class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self,  attn ,place_in_unet):
        key = f"{place_in_unet}_{'cross'}"
        #print("refineattn",attn.size())
        attn_base = self.step_store[key][self.count_attn[key]]
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        #print("attn_base_replace",attn_base_replace.size())
        attn_replace = attn_base_replace.detach() * self.alphas + attn * (1 - self.alphas)
        
            
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None,name=None,word=None,adj_mask=None,word_ng=None):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend,name=name,word=word,adj_mask=adj_mask,word_ng=word_ng)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts[:2], tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])
        
       
    
class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
                local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.equalizer = equalizer.to(device)
        #print("equalize",self.equalizer)
        self.prev_controller = controller


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                  Tuple[float, ...]]):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(1, 77)
    
    for word, val in zip(word_select, values):
        inds = ptp_utils1.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = val
        #print("get_eq_word",word)
        #print("equalizer", equalizer)
    return equalizer

def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.step_atten
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(1, -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    #out = torch.cat(out, dim=0)
    #out = out.sum(0) / out.shape[0]
    #out = attention_store.step_atten
    return out

def aggregate_attention2(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int, prompts):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()

def make_controller(prompts: List[str], is_replace_controller: bool, cross_replace_steps: Dict[str, float], self_replace_steps: float, blend_words=None, equilizer_params=None,name=None,word=None,adj_mask=None,word_ng=None) -> AttentionControlEdit:
    if blend_words is None:
        lb = None
    else:
        lb = LocalBlend(prompts, blend_words,name=name)
    if is_replace_controller:
        controller = AttentionReplace(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, local_blend=lb,name=name,word=word,adj_mask=adj_mask,word_ng=word_ng)
    else:
        controller = AttentionRefine(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, local_blend=lb,name=name,word=word,adj_mask=adj_mask,word_ng=word_ng)
    if equilizer_params is not None:
        eq = get_equalizer(prompts[1], equilizer_params["words"], equilizer_params["values"])
        controller = AttentionReweight(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps,
                                       self_replace_steps=self_replace_steps, equalizer=eq, local_blend=lb, controller=controller)
    return controller


def show_cross_attention(attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = ptp_utils1.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    ptp_utils1.view_images(np.stack(images, axis=0))
    

def show_self_attention_comp(attention_store: AttentionStore, res: int, from_where: List[str],
                        max_com=10, select: int = 0):
    attention_maps = aggregate_attention(attention_store, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    ptp_utils1.view_images(np.concatenate(images, axis=1))
    
def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image


class NullInversion:
    
    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample
    
    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else GUIDANCE_SCALE
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(device)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(NUM_DDIM_STEPS):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents

    def null_optimization(self, latents, num_inner_steps, epsilon):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        bar = tqdm(total=num_inner_steps * NUM_DDIM_STEPS)
        for i in range(NUM_DDIM_STEPS):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2]
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, num_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context)
        bar.close()
        return uncond_embeddings_list
    
    def invert(self, image_path: str, prompt: str, offsets=(0,0,0,0), num_inner_steps=10, early_stop_epsilon=1e-5, verbose=False):
        self.init_prompt(prompt)
        ptp_utils1.register_attention_control(self.model, None)
        image_gt = load_512(image_path, *offsets)
        if verbose:
            print("DDIM inversion...")
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        if verbose:
            print("Null-text optimization...")
        uncond_embeddings = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon)
        return (image_gt, image_rec), ddim_latents[-1], uncond_embeddings
        
    
    def __init__(self, model):
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                  set_alpha_to_one=False)
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(NUM_DDIM_STEPS)
        self.prompt = None
        self.context = None
        

@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    uncond_embeddings=None,
    start_time=50,
    return_type='image'
):
    batch_size = len(prompt)
    ptp_utils1.register_attention_control(model, controller)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
   
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    
    
    max_length = text_input.input_ids.shape[-1]
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None

    latent, latents = ptp_utils1.init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    print(num_inference_steps)
    for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):
        if uncond_embeddings_ is None:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
        latents = ptp_utils1.diffusion_step8(model, controller, latents, context, t, guidance_scale, low_resource=False)
        
    if return_type == 'image':
        image = ptp_utils1.latent2image(model.vae, latents)
    else:
        image = latents
    return image, latent


def renormalize(attn,prompt,normalize_sot=True,normalize_eot=False):
    last_idx = -1
    first_idx = 1 if normalize_sot else 0
    if normalize_eot:
        prompt = prompt
        last_idx = len(tokenizer(prompt)['input_ids']) - 1
    attention_for_text = attn[:,:, :, first_idx:last_idx]
    attention_for_text = attention_for_text*100
    attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)
    return attention_for_text

def renormalize2(attn,prompt,normalize_sot=True,normalize_eot=False):
    last_idx = -1
    first_idx = 1 if normalize_sot else 0
    if normalize_eot:
        prompt = prompt
        last_idx = len(tokenizer(prompt)['input_ids']) - 1
    attention_for_text = attn
    attention_for_text = attention_for_text*100
    attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)
    return attention_for_text


def compute_ca_loss3(attn_maps, masks, word,prompt,name):
    loss = 0.0
    
    #print(word,prompt)
    for obj_idx in range(len(word[1])):
        if  word[1][obj_idx] not in prompt:
            continue  
        adj_idx = ptp_utils1.get_word_inds(prompt, word[1][obj_idx], tokenizer)
        
        
        obj_loss = 0
        for attn_map in attn_maps:
          b, i, j = attn_map.shape
          #print(b,i,j)
          H = W = int(sqrt(i))
          if W==8:
              idx = 0
          elif W==16:
              idx = 1
          elif W==32:
              idx=2
          elif W==64:
              idx=3
          mask = masks[word[0][obj_idx]][idx].squeeze(0)
          n = mask.sum() 
          
          attn_map_ = attn_map
          ca_map_obj = attn_map_[:, :, adj_idx].mean(-1).reshape(b, H, W)
          mask_sum = attn_map_.reshape(b, H, W,-1).sum(dim=-1)
          activation_value = ((ca_map_obj  /mask_sum)* mask).reshape(b, -1).sum(dim=-1)
          activation_value = activation_value/n
          obj_loss += torch.mean((1 - activation_value) ** 2)
        loss+=obj_loss

    
    loss = loss / (len(attn_maps) * (len(word[1])))
  
    #with open(os.path.join(name,"loss.txt"),"a") as file:
            #file.write("loss3:{}\n ".format(loss)) 

    return loss









 


def compute_ca_loss2(attn_maps, masks, word,prompt,name):
    loss = 0.0
    #print(attn_map.size())
    
   # print(word,prompt)
  
    for obj_idx in range(len(word[1])):
          if  word[1][obj_idx] not in prompt:
            continue
          obj_loss = 0
          adj_idx = ptp_utils1.get_word_inds(prompt, word[1][obj_idx], tokenizer)
          adj_idx = adj_idx-1
          
          
          for attn_map in attn_maps:
              #print(attn_map.size())
              h,H, W, j = attn_map.shape
             
              if W==8:
                  idx = 0
              elif W==16:
                  idx = 1
              elif W==32:
                  idx=2
              elif W==64:
                  idx=3
          
              attn_map_ = renormalize(attn_map,prompt) 
              mask = masks[word[0][obj_idx]][idx].squeeze(0)
              ca_map_obj = attn_map_[:, :,:, adj_idx].mean(-1)* mask
      
              smooth_attentions=True
              sigma= 0.5
              kernel_size = 5
              if smooth_attentions:
                  smoothing = GaussianSmoothing(channels=h, kernel_size=kernel_size, sigma=sigma, dim=2).cuda()
                  input = nnf.pad(ca_map_obj.unsqueeze(0), (2, 2, 2, 2),mode='constant', value=0)
                  
                  ca_map_obj = smoothing(input).squeeze(0).reshape(h,-1)
                  #print("before",input.size())
              #print("max",ca_map_obj.max(dim=1)[0].size())
              obj_loss += (1.- ca_map_obj.max(dim=1)[0]).mean()   
          loss+=obj_loss


    loss = loss/(len(attn_maps)*(len(word[1])))
    #with open(os.path.join(name,"loss.txt"),"a") as file:
            #file.write("loss:{}\n ".format(loss)) 

    return loss


def compute_ca_loss1(attn_maps, masks, word,prompt,name):
    loss = 0
     
    for attn_map in attn_maps:
        b, i, j = attn_map.shape
        H = W = int(sqrt(i))
        if W==8:
            idx = 0
        elif W==16:
            idx = 1
        elif W==32:
            idx=2
        elif W==64:
            idx=3
        obj_loss = 0
       
        for obj_idx in range(len(word[1])):
             if  word[1][obj_idx] not in prompt:
               continue             
             adj_idx = ptp_utils1.get_word_inds(prompt, word[1][obj_idx], tokenizer)
             
             mask = masks[word[0][obj_idx]][idx].squeeze(0)
             #mask2 = mask.unsqueeze(-1)
             ca_map_obj = attn_map[:, :, adj_idx].sum(-1).reshape(b, H, W)
             ca_map_sot = attn_map[:, :, [0]].sum(-1).reshape(b, H, W)
             #mask_sum = (attn_map.reshape(b, H, W,-1)* mask2).reshape(b, -1).sum(dim=-1).unsqueeze(-1).unsqueeze(-1)
             #print("mask_sum",mask_sum.size())
             #print("ca_map_obj",ca_map_obj.size())
             #print("mask",mask.size())
             #activation_value = (ca_map_obj * mask /mask_sum).reshape(b, -1).sum(dim=-1)
             
             
             activation_value = (ca_map_obj * mask).reshape(b, -1).sum(dim=-1)/ca_map_obj.reshape(b, -1).sum(dim=-1)
             activation_sot = (ca_map_sot * mask).reshape(b, -1).sum(dim=-1)/ca_map_sot.reshape(b, -1).sum(dim=-1)
             #activation_value = (ca_map_obj * mask).reshape(b, -1).sum(dim=-1)/(attn_map.reshape(b, H, W,-1) * mask2).reshape(b, -1).sum(dim=-1)
             #obj_loss += torch.mean((1 - activation_value) ** 2)
             obj_loss = obj_loss + torch.mean(3- 4 * activation_value)
        loss += obj_loss

    
    loss = loss / (len(word[1]) * len(attn_maps))
  

    return loss





def compute_ca_loss1_2(attn_maps, masks, word,prompt,name):
    loss = 0.0
    attn_map = torch.cat(attn_maps,dim=0)
    idx = 1
    b, i, j = attn_map.shape
    H = W = int(sqrt(i))
    #print("H=",H)
    for obj_idx in range(len(word[1])):
         adj_idx = ptp_utils1.get_word_inds(prompt, word[1][obj_idx], tokenizer)
         mask = masks[word[0][obj_idx]][idx].squeeze(0).float()
         ca_map_obj = attn_map[:, :, adj_idx].sum(-1).reshape(b, H, W)     
         activation_value =  (ca_map_obj * (1-mask)).reshape(b, -1).sum(dim=-1) - (ca_map_obj * mask).reshape(b, -1).sum(dim=-1)
         loss += torch.mean(activation_value)
   
    loss = loss / (len(word[1]))
    return loss

def update_latent_lossadd123_(model, latents1, latents2,  text_embeddings1, text_embeddings2, text_embeddings3, controller, t,i):
    loss1 = torch.tensor(10000)
    loss2 = 0
    loss = 0.0
    loss3 = 0
    
    iteration1 = 0
    iteration2 = 0
    iteration3 = 0
    iteration=0
    #loss1_scale=2.5
    loss1_scale = 2.5
    scale_range=(2. ,.5)
    scale_factor=5.0
    scale_range = np.linspace(scale_range[0], scale_range[1], 50)
    step_size = scale_factor * np.sqrt(scale_range[i])
    loss3_scale  = 2.5
 
    max_iter = 15
    loss3_ngscale = 2.5
    loss1_ngscale = 5.0
    loss2_ngscale = 2.0
    loss1_max_iter=8
    loss2_max_iter=8
    loss3_max_iter=8
    flag1 = False
    flag2 = False
    flag3 = False
    sigma = sqrt((1-model.scheduler.alphas_cumprod[t])/model.scheduler.alphas_cumprod[t])
    loss1_threshold=2.0
    loss2_threshold = {0: 0.15 , 9:0.1, 19: 0.1}  
    if i<10:
     
      loss3_threshold=0.955
      #loss3_threshold=0.91
    elif 10<=i<20:
     
      loss3_threshold=0.96
      #loss3_threshold=0.92
    else:
      
      loss3_threshold=0.97
      #loss3_threshold=0.93

    latents2 = latents2.requires_grad_(True)
    
    mask = torch.zeros(1,1,64,64).cuda()
    for k in range(len(controller.word[1])):
      if k==0 or (k>=1 and controller.word[0][k]!=controller.word[0][k-1] ):
        mask += controller.adj_mask[controller.word[0][k]][3]
    
     
    iteration=0
    
    noise_prediction = model.unet(latents1, t, encoder_hidden_states=text_embeddings1)["sample"]
    noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings2)["sample"]   
    atten = controller.step_atten['up_cross'][0:3]+controller.step_atten['down_cross'][4:6]
    loss1 = compute_ca_loss1(atten, controller.adj_mask, controller.word,controller.prompt, controller.name)      
    loss3 = compute_ca_loss3(atten, controller.adj_mask,controller.word, controller.prompt,controller.name)
    loss = loss1+loss3
    loss_threshold = loss1_threshold + loss3_threshold
    if i in loss2_threshold.keys():  
      atten2 = [a.reshape(8,16,16,77) for a in atten]
      loss2 = compute_ca_loss2(atten2, controller.adj_mask,controller.word, controller.prompt,controller.name)
      loss = loss+loss2
      loss_threshold += max(0, 1. - loss2_threshold[i]) 
    print("loss_threshold",loss_threshold)
    print("first_loss",loss.item())
    if i<25:
        
        while iteration < max_iter and loss>loss_threshold:
                iteration+=1
                atten = controller.step_atten['up_cross'][0:3]+controller.step_atten['down_cross'][4:6]
                loss1 = compute_ca_loss1(atten, controller.adj_mask, controller.word,controller.prompt, controller.name) * loss1_scale     
                loss3 = compute_ca_loss3(atten, controller.adj_mask,controller.word, controller.prompt,controller.name) * loss3_scale * sigma 
                loss = loss1+loss3
                
                if i in loss2_threshold.keys():  
                  atten2 = [a.reshape(8,16,16,77) for a in atten]
                  loss2 = compute_ca_loss2(atten2, controller.adj_mask,controller.word, controller.prompt,controller.name)*step_size
                  loss = loss+loss2
                
                #print("total_loss",loss.item())
                #negative
                '''if controller.word_ng is not None:
                    #noise_prediction = model.unet(latents1, t, encoder_hidden_states=text_embeddings1)["sample"]
                    #noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings3)["sample"]                    
                    atten = controller.step_atten['up_cross'][0:3]+controller.step_atten['down_cross'][4:6]
                    loss1_ng = compute_ca_loss1(atten, controller.adj_mask,controller.word_ng,controller.ng_prompt,controller.name)                
                    loss1 =  loss1-1.0* loss1_ngscale * loss1_ng
                    print("loss1_ng",loss1.item())'''

                '''if controller.word_ng is not None:
                  #noise_prediction = model.unet(latents1, t, encoder_hidden_states=text_embeddings1)["sample"]
                  #noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings3)["sample"]   
                  atten = controller.step_atten['up_cross'][0:3]+controller.step_atten['down_cross'][4:6]
                  loss3_ng = compute_ca_loss3(atten, controller.adj_mask,controller.word_ng,controller.ng_prompt,controller.name)
                  loss3 = loss3 - 1.0 * loss3_ngscale * loss3_ng        
                  print("loss3_ng",loss3.item())'''
              
                
               
                
                
                grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents2], retain_graph=True)[0]
                latent_unmask = latents2.detach().clone()* (1-mask)
                latent_mask = latents2 * mask
                latent_mask = latent_mask -  grad_cond * mask
                latents2 = latent_unmask + latent_mask
                

                #with torch.no_grad():
                noise_prediction = model.unet(latents1.detach(), t, encoder_hidden_states=text_embeddings1)["sample"]
                noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings2)["sample"]   
                atten = controller.step_atten['up_cross'][0:3]+controller.step_atten['down_cross'][4:6]
                loss1 = compute_ca_loss1(atten, controller.adj_mask, controller.word,controller.prompt, controller.name)
                loss3 = compute_ca_loss3(atten, controller.adj_mask,controller.word, controller.prompt,controller.name) 
                loss = loss1+loss3     
                if i in loss2_threshold.keys():  
                  atten2 = [a.reshape(8,16,16,77) for a in atten]
                  loss2 = compute_ca_loss2(atten2, controller.adj_mask, controller.word,controller.prompt,controller.name)
                  loss = loss + loss2
                print(loss)
                if iteration >= max_iter:
                  break
       
       
        
    return latents2.detach()      



def update_latent_lossadd123_wo_v(model, latents1, latents2,  text_embeddings1, text_embeddings2, text_embeddings3, controller, t,i):
    loss1 = torch.tensor(10000)
    loss2 = 0
    loss = 0.0
    loss3 = 0
    
    iteration1 = 0
    iteration2 = 0
    iteration3 = 0
    iteration=0
    sigma = sqrt((1-model.scheduler.alphas_cumprod[t])/model.scheduler.alphas_cumprod[t])
  
    loss1_scale=2.5
    scale_range=(2. ,.5)
    loss3_scale  = 2.5
   
    ##
    max_iter = 15
    #loss3_ngscale = 0.5
    loss3_ngscale=2.5
    #loss1_ngscale = 2.5
    loss1_ngscale=2.5
    loss2_ngscale = 2.0
    loss1_max_iter=8
    loss2_max_iter=8
    loss3_max_iter=8
   
    
    print("sigma",sigma)
    loss1_threshold=2.0  
    loss2_threshold = {0: 0.15 , 9:0.1, 19: 0.1}  
    if i<10:
     
      loss3_threshold=0.95
      #loss3_threshold=0.91
    elif 10<=i<20:
     
      loss3_threshold=0.96
      #loss3_threshold=0.92
    else:
      
      loss3_threshold=0.97
      #loss3_threshold=0.93

    latents2 = latents2.requires_grad_(True)
    
    mask = torch.zeros(1,1,64,64).cuda()
    for k in range(len(controller.word[1])):
      if k==0 or (k>=1 and controller.word[0][k]!=controller.word[0][k-1] ):
        mask += controller.adj_mask[controller.word[0][k]][3]
    
     
    iteration=0
    
    noise_prediction = model.unet(latents1, t, encoder_hidden_states=text_embeddings1)["sample"]
    noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings2)["sample"]   
          
    # update latents with guidance
    #atten = controller.step_atten['up_cross'][0:3]+controller.step_atten['down_cross'][4:6]
    #loss1 = compute_ca_loss1(atten, controller.adj_mask, controller.word,controller.prompt,controller.name)
    #loss1 = loss1.requires_grad_(True)
    #controller.get_attn_change(atten,"before")
    #print("loss1_first",loss1.item())
    if i<25:#
        
        while iteration < max_iter:
                iteration+=1
                #noise_prediction = model.unet(latents1, t, encoder_hidden_states=text_embeddings1)["sample"]
                #noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings2)["sample"]                 
                #positive
                atten = controller.step_atten['up_cross'][0:3]+controller.step_atten['down_cross'][4:6]
                
                loss1 = compute_ca_loss1(atten, controller.adj_mask, controller.word,controller.prompt, controller.name) * loss1_scale
                loss3 = compute_ca_loss3(atten, controller.adj_mask,controller.word, controller.prompt,controller.name) * loss3_scale
                #print("loss1:",loss1.item())
                
                #negative
                '''if controller.word_ng is not None:
                    noise_prediction = model.unet(latents1, t, encoder_hidden_states=text_embeddings1)["sample"]
                    noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings3)["sample"]                    
                    atten = controller.step_atten['up_cross'][0:3]+controller.step_atten['down_cross'][4:6]
                    loss1_ng = compute_ca_loss1(atten, controller.adj_mask,controller.word_ng,controller.ng_prompt,controller.name)   
                    loss3_ng = compute_ca_loss3(atten, controller.adj_mask,controller.word_ng,controller.ng_prompt,controller.name)             
                    loss1 =  loss1-1.0* loss1_ngscale * loss1_ng
                    loss3 = loss3 - 1.0 * loss3_ngscale * loss3_ng       
                    print("loss3",loss3.item()) '''
                
                loss = loss1+loss3
                
                '''if i in loss2_threshold.keys():  
                    atten2 = [a.reshape(8,16,16,77) for a in atten]
                    loss2 = compute_ca_loss2(atten2, controller.adj_mask,controller.word, controller.prompt,controller.name)*step_size    
                    #print("Loss2:",loss2.item())  
                    if controller.word_ng is not None:
                      loss2_ng = compute_ca_loss2(atten2, controller.adj_mask, controller.word_ng,controller.ng_prompt,controller.name)
                      print("loss2_ng",loss2.item())
                      loss2 =  loss2 - 1.0 *  loss2_ngscale * loss2_ng       
              
                    loss =  loss+loss2 '''                  
               
                print("total_loss",loss.item())
                grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents2], retain_graph=True)[0]
                latent_unmask = latents2.detach().clone()* (1-mask)
                latent_mask = latents2 * mask
                latent_mask = latent_mask -  grad_cond * mask
                latents2 = latent_unmask + latent_mask
                

                #with torch.no_grad():
                noise_prediction = model.unet(latents1.detach(), t, encoder_hidden_states=text_embeddings1)["sample"]
                noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings2)["sample"]   
                atten = controller.step_atten['up_cross'][0:3]+controller.step_atten['down_cross'][4:6]
                loss1 = compute_ca_loss1(atten, controller.adj_mask, controller.word,controller.prompt, controller.name) * loss1_scale  
                loss3 = compute_ca_loss3(atten, controller.adj_mask,controller.word, controller.prompt,controller.name)* loss3_scale         
  
                loss = loss1 + loss3
                if iteration >= max_iter:
                  break
       
       
        
    return latents2.detach()      



def update_latent_lossadd123(model, latents1, latents2,  text_embeddings1, text_embeddings2, text_embeddings3, controller, t,i):
    loss1 = torch.tensor(10000)
    loss2 = 0
    loss = 0.0
    loss3 = 0
    
    iteration1 = 0
    iteration2 = 0
    iteration3 = 0
    iteration=0
    #loss1_scale=2.5
    loss1_scale = 2.5
    scale_range=(2. ,.5)
    scale_factor=5.0
    scale_range = np.linspace(scale_range[0], scale_range[1], 50)
    step_size = scale_factor * np.sqrt(scale_range[i])
    loss3_scale  = 2.5
 
    max_iter = 15
    loss3_ngscale = 2.5
    loss1_ngscale = 5.0
    loss2_ngscale = 2.0
    loss1_max_iter=15
    loss2_max_iter=15
    loss3_max_iter=15
    flag1 = False
    flag2 = False
    flag3 = False
    sigma = sqrt((1-model.scheduler.alphas_cumprod[t])/model.scheduler.alphas_cumprod[t])
    loss1_threshold=2.0   
    loss2_threshold = {0: 0.15 , 9:0.1, 19: 0.1}  
    if i<10:
     
      loss3_threshold=0.95
      #loss3_threshold=0.91
    elif 10<=i<20:
     
      loss3_threshold=0.96
      #loss3_threshold=0.92
    else:
      
      loss3_threshold=0.97
      #loss3_threshold=0.93

    latents2 = latents2.requires_grad_(True)
    
    mask = torch.zeros(1,1,64,64).cuda()
    for k in range(len(controller.word[1])):
      if k==0 or (k>=1 and controller.word[0][k]!=controller.word[0][k-1] ):
        mask += controller.adj_mask[controller.word[0][k]][3]
    
     
    iteration=0
    
    noise_prediction = model.unet(latents1, t, encoder_hidden_states=text_embeddings1)["sample"]
    noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings2)["sample"]   
          
    # update latents with guidance
    #atten = controller.step_atten['up_cross'][0:3]+controller.step_atten['down_cross'][4:6]
    #loss1 = compute_ca_loss1(atten, controller.adj_mask, controller.word,controller.prompt,controller.name)
    #loss1 = loss1.requires_grad_(True)
    #controller.get_attn_change(atten,"before")
    #print("loss1_first",loss1.item())
    if i<25:
        
        while iteration < max_iter:
                iteration+=1
                #noise_prediction = model.unet(latents1, t, encoder_hidden_states=text_embeddings1)["sample"]
                #noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings2)["sample"]                 
                #positive
                atten = controller.step_atten['up_cross'][0:3]+controller.step_atten['down_cross'][4:6]
                
                loss1 = compute_ca_loss1(atten, controller.adj_mask, controller.word,controller.prompt, controller.name) * loss1_scale
                #print("loss1:",loss1.item())
                
                #negative
                '''if controller.word_ng is not None:
                    #noise_prediction = model.unet(latents1, t, encoder_hidden_states=text_embeddings1)["sample"]
                    #noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings3)["sample"]                    
                    atten = controller.step_atten['up_cross'][0:3]+controller.step_atten['down_cross'][4:6]
                    loss1_ng = compute_ca_loss1(atten, controller.adj_mask,controller.word_ng,controller.ng_prompt,controller.name)                
                    loss1 =  loss1-1.0* loss1_ngscale * loss1_ng
                    print("loss1_ng",loss1.item())'''
                loss = loss1
                
                loss3 = compute_ca_loss3(atten, controller.adj_mask,controller.word, controller.prompt,controller.name) * loss3_scale * sigma
                #print("loss3:",loss3.item())
                '''if controller.word_ng is not None:
                  #noise_prediction = model.unet(latents1, t, encoder_hidden_states=text_embeddings1)["sample"]
                  #noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings3)["sample"]   
                  atten = controller.step_atten['up_cross'][0:3]+controller.step_atten['down_cross'][4:6]
                  loss3_ng = compute_ca_loss3(atten, controller.adj_mask,controller.word_ng,controller.ng_prompt,controller.name)
                  loss3 = loss3 - 1.0 * loss3_ngscale * loss3_ng        
                  print("loss3_ng",loss3.item())'''
                loss =  loss+loss3
                
                if i in loss2_threshold.keys():  
                    atten2 = [a.reshape(8,16,16,77) for a in atten]
                    loss2 = compute_ca_loss2(atten2, controller.adj_mask,controller.word, controller.prompt,controller.name)*step_size    
                    #print("Loss2:",loss2.item())  
                    '''if controller.word_ng is not None:
                      loss2_ng = compute_ca_loss2(atten2, controller.adj_mask, controller.word_ng,controller.ng_prompt,controller.name)
                      print("loss2_ng",loss2.item())
                      loss2 =  loss2 - 1.0 *  loss2_ngscale * loss2_ng'''       
              
                    loss =  loss+loss2                      
               
                print("total_loss",loss.item())
                grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents2], retain_graph=True)[0]
                latent_unmask = latents2.detach().clone()* (1-mask)
                latent_mask = latents2 * mask
                latent_mask = latent_mask -  grad_cond * mask
                latents2 = latent_unmask + latent_mask
                

                #with torch.no_grad():
                noise_prediction = model.unet(latents1.detach(), t, encoder_hidden_states=text_embeddings1)["sample"]
                noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings2)["sample"]   
                atten = controller.step_atten['up_cross'][0:3]+controller.step_atten['down_cross'][4:6]
                loss1 = compute_ca_loss1(atten, controller.adj_mask, controller.word,controller.prompt, controller.name) * loss1_scale  
                loss3 = compute_ca_loss3(atten, controller.adj_mask,controller.word, controller.prompt,controller.name)* loss3_scale * sigma           
                atten = aggregate_attention( attention_store=controller,\
                                                res=16,\
                                                from_where=("up", "down"),\
                                                is_cross=True,\
                                                select=0)
                loss2 = compute_ca_loss2(atten, controller.adj_mask, controller.word,controller.prompt,controller.name)*step_size  
                loss = loss1 + loss2 + loss3
                if iteration >= max_iter:
                  break
       
       
        
    return latents2.detach()      


def update_latent4(model, latents1, latents2,  text_embeddings1, text_embeddings2, text_embeddings3, controller, t,i):
    loss1 = torch.tensor(10000)
    loss2 = 0
    loss = 0
    loss3 = 0
    
    iteration1 = 0
    iteration2 = 0
    iteration3 = 0
    
    #loss1_scale=2.5
    loss1_scale = 2.5
    scale_range=(2. ,.5)
    scale_factor=5.0
    scale_range = np.linspace(scale_range[0], scale_range[1], 50)
    step_size = scale_factor * np.sqrt(scale_range[i])
    loss3_scale  = 2.5
 
    
    loss3_ngscale = 2.5
    loss1_ngscale = 5.0
    loss2_ngscale = 2.0
    loss1_max_iter=15
    loss2_max_iter=15
    loss3_max_iter=15
    flag1 = False
    flag2 = False
    flag3 = False
    
    loss1_threshold=2.0   
    loss2_threshold = {0: 0.15 , 9:0.1, 19: 0.1}  
    if i<10:
     
      loss3_threshold=0.95
      #loss3_threshold=0.91
    elif 10<=i<20:
     
      loss3_threshold=0.96
      #loss3_threshold=0.92
    else:
      
      loss3_threshold=0.97
      #loss3_threshold=0.93

    latents2 = latents2.requires_grad_(True)
    
    mask = torch.zeros(1,1,64,64).cuda()
    for k in range(len(controller.word[1])):
      if k==0 or (k>=1 and controller.word[0][k]!=controller.word[0][k-1] ):
        mask += controller.adj_mask[controller.word[0][k]][3]
    
     
 
    
    noise_prediction = model.unet(latents1, t, encoder_hidden_states=text_embeddings1)["sample"]
    noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings2)["sample"]   
          
    # update latents with guidance
    atten = controller.step_atten['up_cross'][0:3]+controller.step_atten['down_cross'][4:6]
    loss1 = compute_ca_loss1(atten, controller.adj_mask, controller.word,controller.prompt,controller.name)
    #loss1 = loss1.requires_grad_(True)
    #controller.get_attn_change(atten,"before")
    print("loss1_first",loss1.item())
    if i<25:
        
        while loss1.item()  > loss1_threshold and iteration1 < loss1_max_iter:
              
                #positive
                atten = controller.step_atten['up_cross'][0:3]+controller.step_atten['down_cross'][4:6]
                loss1 = compute_ca_loss1(atten, controller.adj_mask, controller.word,controller.prompt, controller.name)
                loss = loss1*loss1_scale
                print("loss1:",loss1.item())
                
                #negative
                if controller.word_ng is not None:
                    noise_prediction = model.unet(latents1, t, encoder_hidden_states=text_embeddings1)["sample"]
                    noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings3)["sample"]                    
                    atten = controller.step_atten['up_cross'][0:3]+controller.step_atten['down_cross'][4:6]
                    loss1 = compute_ca_loss1(atten, controller.adj_mask,controller.word_ng,controller.ng_prompt,controller.name)                
                    loss =  loss-1.0* loss1_ngscale *  loss1
                    print("loss1_ng",loss1.item())
                grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents2])[0]
                sigma = sqrt((1-model.scheduler.alphas_cumprod[t])/model.scheduler.alphas_cumprod[t])
                latent_unmask = latents2.detach().clone()* (1-mask)
                latent_mask = latents2 * mask
                latent_mask = latent_mask - grad_cond  * mask
                latents2 = latent_unmask + latent_mask
                iteration1 += 1
                torch.cuda.empty_cache() 
                
                noise_prediction = model.unet(latents1.detach(), t, encoder_hidden_states=text_embeddings1)["sample"]
                noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings2)["sample"]   
                atten = controller.step_atten['up_cross'][0:3]+controller.step_atten['down_cross'][4:6]
                loss1 = compute_ca_loss1(atten, controller.adj_mask, controller.word,controller.prompt,controller.name) 
                
            
      
       
        loss3 = compute_ca_loss3(atten, controller.adj_mask,controller.word, controller.prompt,controller.name)
        print("firstloss3",loss3.item())
        while loss3.item()  > loss3_threshold and iteration3 < loss3_max_iter:
       
          
          # positive
          atten = controller.step_atten['up_cross'][0:3]+controller.step_atten['down_cross'][4:6]
          loss3 = compute_ca_loss3(atten, controller.adj_mask,controller.word, controller.prompt,controller.name)
          loss = loss3*loss3_scale
          print("loss3:",loss3.item())
          
          if controller.word_ng is not None:
            noise_prediction = model.unet(latents1, t, encoder_hidden_states=text_embeddings1)["sample"]
            noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings3)["sample"]   
            atten = controller.step_atten['up_cross'][0:3]+controller.step_atten['down_cross'][4:6]
            loss3 = compute_ca_loss3(atten, controller.adj_mask,controller.word_ng,controller.ng_prompt,controller.name)
            loss = loss -1.0 * loss3_ngscale * loss3        
            print("loss3_ng",loss3.item())
                          
          grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents2])[0]
          sigma = sqrt((1-model.scheduler.alphas_cumprod[t])/model.scheduler.alphas_cumprod[t])
          latent_unmask = latents2.detach().clone()* (1-mask)
          latent_mask = latents2 * mask
          latent_mask = latent_mask - grad_cond * sigma * mask
          latents2 = latent_unmask + latent_mask
          iteration3 += 1
          torch.cuda.empty_cache()   
          
          noise_prediction = model.unet(latents1, t, encoder_hidden_states=text_embeddings1)["sample"]
          noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings2)["sample"]   
          # update latents with guidance
          atten = controller.step_atten['up_cross'][0:3]+controller.step_atten['down_cross'][4:6]
          loss3 = compute_ca_loss3(atten, controller.adj_mask,controller.word, controller.prompt,controller.name)       
         
        if flag3: 
          controller.get_attn_change(atten,"loss3")

        
    if i in loss2_threshold.keys():  
        atten = aggregate_attention( attention_store=controller,\
                                                res=16,\
                                                from_where=("up", "down"),\
                                                is_cross=True,\
                                                select=0)  
        loss2 = compute_ca_loss2(atten, controller.adj_mask,controller.word, controller.prompt,controller.name)         
        print("first Loss2:",loss2.item()) 
        
        iteration2=0  
        target_loss = max(0, 1. - loss2_threshold[i])
        while loss2 > target_loss:
           
            iteration2 += 1
            noise_prediction = model.unet(latents1, t, encoder_hidden_states=text_embeddings1)["sample"]
            noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings2)["sample"]   

            # positive
            atten = aggregate_attention( attention_store=controller,\
                                                  res=16,\
                                                  from_where=("up", "down"),\
                                                  is_cross=True,\
                                                  select=0)

            loss2 = compute_ca_loss2(atten, controller.adj_mask,controller.word, controller.prompt,controller.name)
            loss = loss2                    
            print("Loss2:",loss2.item())  
            
            if controller.word_ng is not None:
              noise_prediction = model.unet(latents1, t, encoder_hidden_states=text_embeddings1)["sample"]
              noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings3)["sample"]               
              atten = aggregate_attention( attention_store=controller,\
                                                  res=16,\
                                                  from_where=("up", "down"),\
                                                  is_cross=True,\
                                                  select=0)
      
              loss2 = compute_ca_loss2(atten, controller.adj_mask, controller.word_ng,controller.ng_prompt,controller.name)
              print("loss2_ng",loss2.item())
            loss =  loss -1.0 *  loss2_ngscale * loss2                              
            
            
            grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents2], retain_graph=True)[0]
            latent_unmask = latents2.detach().clone()* (1-mask)
            latent_mask = latents2 * mask
            latent_mask = latent_mask - step_size * grad_cond * mask
            latents2 = latent_unmask + latent_mask
                

            with torch.no_grad():
                 noise_prediction = model.unet(latents1.detach(), t, encoder_hidden_states=text_embeddings1)["sample"]
                 noise_prediction = model.unet(latents2.detach(), t, encoder_hidden_states=text_embeddings2)["sample"]   
                 atten = aggregate_attention( attention_store=controller,\
                                                  res=16,\
                                                  from_where=("up", "down"),\
                                                  is_cross=True,\
                                                  select=0)
                 loss2 = compute_ca_loss2(atten, controller.adj_mask, controller.word,controller.prompt,controller.name)
            if iteration2 >= loss2_max_iter:
                break
       
       
        
    return latents2.detach()      





def update_latent2_loss23(model, latents1, latents2,  text_embeddings1, text_embeddings2, text_embeddings3, controller, t,i):
    loss1 = torch.tensor(10000)
    loss2 = 0
    loss = 0
    loss3 = 0
    
    iteration1 = 0
    iteration2 = 0
    iteration3 = 0
    
    #loss1_scale=2.5
    loss1_scale = 2.5
    scale_range=(2. ,.5)
    scale_factor=5.0
    scale_range = np.linspace(scale_range[0], scale_range[1], 50)
    step_size = scale_factor * np.sqrt(scale_range[i])
    loss3_scale  = 2.5
    

    
    loss3_ngscale = 5.5
    loss1_ngscale = 5.0
    loss2_ngscale = 2.0
    
    loss1_max_iter=15
    loss2_max_iter=15
    loss3_max_iter=15
    flag1 = False
    flag2 = False
    flag3 = False
    
    loss1_threshold=2.0   
    loss2_threshold = {0: 0.15 , 9:0.1, 19: 0.1}   
    if i<10:
      #loss3_scale=3.5
      loss3_threshold=0.95
      #loss3_threshold=0.91
    elif 10<=i<20:
      #loss3_scale=3.5
      loss3_threshold=0.96
      #loss3_threshold=0.92
    else:
      #loss3_scale=3.5
      loss3_threshold=0.97
      #loss3_threshold=0.93

    latents2 = latents2.requires_grad_(True)
    
    mask = torch.zeros(1,1,64,64).cuda()
    for k in range(len(controller.word[1])):
      if k==0 or (k>=1 and controller.word[0][k]!=controller.word[0][k-1] ):
        mask += controller.adj_mask[controller.word[0][k]][3]
    
     
 
    
    noise_prediction = model.unet(latents1, t, encoder_hidden_states=text_embeddings1)["sample"]
    noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings2)["sample"]   
          
    # update latents with guidance
    atten = controller.step_atten['up_cross'][0:3]+controller.step_atten['down_cross'][4:6]
    loss1 = compute_ca_loss1(atten, controller.adj_mask, controller.word,controller.prompt,controller.name)
    #loss1 = loss1.requires_grad_(True)
    #controller.get_attn_change(atten,"before")
    print("loss1_first",loss1.item())
    if i<25:
        
        while loss1.item()  > loss1_threshold and iteration1 < loss1_max_iter:
              
                #positive
                atten = controller.step_atten['up_cross'][0:3]+controller.step_atten['down_cross'][4:6]
                loss1 = compute_ca_loss1(atten, controller.adj_mask, controller.word,controller.prompt, controller.name)
                loss = loss1*loss1_scale
                print("loss1:",loss1.item())
                
                #negative
                if controller.word_ng is not None:
                    noise_prediction = model.unet(latents1, t, encoder_hidden_states=text_embeddings1)["sample"]
                    noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings3)["sample"]                    
                    atten = controller.step_atten['up_cross'][0:3]+controller.step_atten['down_cross'][4:6]
                    loss1 = compute_ca_loss1(atten, controller.adj_mask,controller.word_ng,controller.ng_prompt,controller.name)                
                    loss =  loss-1.0* loss1_ngscale *  loss1
                    print("loss1_ng",loss1.item())
                grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents2])[0]
                sigma = sqrt((1-model.scheduler.alphas_cumprod[t])/model.scheduler.alphas_cumprod[t])
                latent_unmask = latents2.detach().clone()* (1-mask)
                latent_mask = latents2 * mask
                latent_mask = latent_mask - grad_cond  * mask
                latents2 = latent_unmask + latent_mask
                iteration1 += 1
                torch.cuda.empty_cache() 
                
                noise_prediction = model.unet(latents1.detach(), t, encoder_hidden_states=text_embeddings1)["sample"]
                noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings2)["sample"]   
                atten = controller.step_atten['up_cross'][0:3]+controller.step_atten['down_cross'][4:6]
                loss1 = compute_ca_loss1(atten, controller.adj_mask, controller.word,controller.prompt,controller.name) 
                
            
      
       
        loss3 = compute_ca_loss3(atten, controller.adj_mask,controller.word, controller.prompt,controller.name)
        print("firstloss3",loss3.item())
        while loss3.item()  > loss3_threshold and iteration3 < loss3_max_iter:
       
          
          # positive
          atten = controller.step_atten['up_cross'][0:3]+controller.step_atten['down_cross'][4:6]
          loss3 = compute_ca_loss3(atten, controller.adj_mask,controller.word, controller.prompt,controller.name)
          loss = loss3*loss3_scale
          print("loss3:",loss3.item())
          
          if controller.word_ng is not None:
            noise_prediction = model.unet(latents1, t, encoder_hidden_states=text_embeddings1)["sample"]
            noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings3)["sample"]   
            atten = controller.step_atten['up_cross'][0:3]+controller.step_atten['down_cross'][4:6]
            loss3 = compute_ca_loss3(atten, controller.adj_mask,controller.word_ng,controller.ng_prompt,controller.name)
            loss =  loss-1.0 * loss3_ngscale * loss3        
            print("loss3_ng",loss3.item())
                          
          grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents2])[0]
          sigma = sqrt((1-model.scheduler.alphas_cumprod[t])/model.scheduler.alphas_cumprod[t])
          latent_unmask = latents2.detach().clone()* (1-mask)
          latent_mask = latents2 * mask
          latent_mask = latent_mask - grad_cond * sigma * mask
          latents2 = latent_unmask + latent_mask
          iteration3 += 1
          torch.cuda.empty_cache()   
          
          noise_prediction = model.unet(latents1, t, encoder_hidden_states=text_embeddings1)["sample"]
          noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings2)["sample"]   
          # update latents with guidance
          atten = controller.step_atten['up_cross'][0:3]+controller.step_atten['down_cross'][4:6]
          loss3 = compute_ca_loss3(atten, controller.adj_mask,controller.word, controller.prompt,controller.name)       
         
        if flag3: 
          controller.get_attn_change(atten,"loss3")       
        
    return latents2.detach()  


def update_latent2_loss13(model, latents1, latents2,  text_embeddings1, text_embeddings2, text_embeddings3, controller, t,i):
    loss1 = torch.tensor(10000)
    loss2 = 0
    loss = 0
    loss3 = 0
    
    iteration1 = 0
    iteration2 = 0
    iteration3 = 0
    
   
    loss1_scale = 2.5
    scale_range=(2. ,.5)
    scale_factor=5.0
    scale_range = np.linspace(scale_range[0], scale_range[1], 50)
    step_size = scale_factor * np.sqrt(scale_range[i])
    loss3_scale  = 2.5

   
    
    loss3_ngscale = 5.5
    loss1_ngscale = 5.0
    loss2_ngscale = 2.0
    
    loss1_max_iter=15
    loss2_max_iter=15
    loss3_max_iter=15
    
    flag1 = False
    flag2 = False
    flag3 = False
    
    loss1_threshold=2.0   
    loss2_threshold = {0: 0.15 , 9:0.1, 19: 0.1}   
    if i<10:
      #loss3_scale=3.5
      loss3_threshold=0.95
      #loss3_threshold=0.91
    elif 10<=i<20:
      #loss3_scale=3.5
      loss3_threshold=0.96
      #loss3_threshold=0.92
    else:
      #loss3_scale=3.5
      loss3_threshold=0.97
      #loss3_threshold=0.93

    latents2 = latents2.requires_grad_(True)
    
    mask = torch.zeros(1,1,64,64).cuda()
    for k in range(len(controller.word[1])):
      if k==0 or (k>=1 and controller.word[0][k]!=controller.word[0][k-1] ):
        mask += controller.adj_mask[controller.word[0][k]][3]
    
     
 
    
    noise_prediction = model.unet(latents1, t, encoder_hidden_states=text_embeddings1)["sample"]
    noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings2)["sample"]   
          
    # update latents with guidance
    atten = controller.step_atten['up_cross'][0:3]+controller.step_atten['down_cross'][4:6]
    loss1 = compute_ca_loss1(atten, controller.adj_mask, controller.word,controller.prompt,controller.name)
    #loss1 = loss1.requires_grad_(True)
    #controller.get_attn_change(atten,"before")
    print("loss1_first",loss1.item())
    if i<25:
        
        while loss1.item()  > loss1_threshold and iteration1 < loss1_max_iter:
              
                #positive
                atten = controller.step_atten['up_cross'][0:3]+controller.step_atten['down_cross'][4:6]
                loss1 = compute_ca_loss1(atten, controller.adj_mask, controller.word,controller.prompt, controller.name)
                loss = loss1*loss1_scale
                print("loss1:",loss1.item())
                
                #negative
                if controller.word_ng is not None:
                    noise_prediction = model.unet(latents1, t, encoder_hidden_states=text_embeddings1)["sample"]
                    noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings3)["sample"]                    
                    atten = controller.step_atten['up_cross'][0:3]+controller.step_atten['down_cross'][4:6]
                    loss1 = compute_ca_loss1(atten, controller.adj_mask,controller.word_ng,controller.ng_prompt,controller.name)                
                    loss =  loss - 1.0* loss1_ngscale *  loss1
                    print("loss1_ng",loss1.item())
                grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents2])[0]
                sigma = sqrt((1-model.scheduler.alphas_cumprod[t])/model.scheduler.alphas_cumprod[t])
                latent_unmask = latents2.detach().clone()* (1-mask)
                latent_mask = latents2 * mask
                latent_mask = latent_mask - grad_cond  * mask
                latents2 = latent_unmask + latent_mask
                iteration1 += 1
                torch.cuda.empty_cache() 
                
                noise_prediction = model.unet(latents1.detach(), t, encoder_hidden_states=text_embeddings1)["sample"]
                noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings2)["sample"]   
                atten = controller.step_atten['up_cross'][0:3]+controller.step_atten['down_cross'][4:6]
                loss1 = compute_ca_loss1(atten, controller.adj_mask, controller.word,controller.prompt,controller.name) 
                
            
      
       
        loss3 = compute_ca_loss3(atten, controller.adj_mask,controller.word, controller.prompt,controller.name)
        print("firstloss3",loss3.item())
    

        
    if i in loss2_threshold.keys():  
        atten = aggregate_attention( attention_store=controller,\
                                                res=16,\
                                                from_where=("up", "down"),\
                                                is_cross=True,\
                                                select=0)  
        loss2 = compute_ca_loss2(atten, controller.adj_mask,controller.word, controller.prompt,controller.name)         
        print("first Loss2:",loss2.item()) 
        
        iteration2=0  
        target_loss = max(0, 1. - loss2_threshold[i])
        while loss2 > target_loss:
           
            iteration2 += 1
            noise_prediction = model.unet(latents1, t, encoder_hidden_states=text_embeddings1)["sample"]
            noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings2)["sample"]   

            # positive
            atten = aggregate_attention( attention_store=controller,\
                                                  res=16,\
                                                  from_where=("up", "down"),\
                                                  is_cross=True,\
                                                  select=0)

            loss2 = compute_ca_loss2(atten, controller.adj_mask,controller.word, controller.prompt,controller.name)
            #loss = loss2                    
            print("Loss2:",loss2.item())  
            
            if controller.word_ng is not None:
              noise_prediction = model.unet(latents1, t, encoder_hidden_states=text_embeddings1)["sample"]
              noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings3)["sample"]               
              atten = aggregate_attention( attention_store=controller,\
                                                  res=16,\
                                                  from_where=("up", "down"),\
                                                  is_cross=True,\
                                                  select=0)
      
              loss2 = compute_ca_loss2(atten, controller.adj_mask, controller.word_ng,controller.ng_prompt,controller.name)
              print("loss2_ng",loss2.item())
            loss =   -1.0 *  loss2_ngscale * loss2                              
            
            
            grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents2], retain_graph=True)[0]
            latent_unmask = latents2.detach().clone()* (1-mask)
            latent_mask = latents2 * mask
            latent_mask = latent_mask - step_size * grad_cond * mask
            latents2 = latent_unmask + latent_mask
                

            with torch.no_grad():
                 noise_prediction = model.unet(latents1.detach(), t, encoder_hidden_states=text_embeddings1)["sample"]
                 noise_prediction = model.unet(latents2.detach(), t, encoder_hidden_states=text_embeddings2)["sample"]   
                 atten = aggregate_attention( attention_store=controller,\
                                                  res=16,\
                                                  from_where=("up", "down"),\
                                                  is_cross=True,\
                                                  select=0)
                 loss2 = compute_ca_loss2(atten, controller.adj_mask, controller.word,controller.prompt,controller.name)
            if iteration2 >= loss2_max_iter:
                break
       
       
        
    return latents2.detach()  
    
    

def update_latent2_loss12(model, latents1, latents2, text_embeddings1, text_embeddings2, text_embeddings3,controller, t, i):
    latents2 = latents2.requires_grad_(True)
  
    loss2 = 0
    loss = 0
    loss3 = 0
    

    iteration2 = 0
    iteration3 = 0

    loss3_ngscale= 2.5
    loss2_ngscale = 1.0
    
    loss3_scale = 2.5
    scale_range = (2., .5)
    scale_factor = 5.0
    scale_range = np.linspace(scale_range[0], scale_range[1], 50)
    step_size = scale_factor * np.sqrt(scale_range[i])
  


    loss2_max_iter = 15
    loss3_max_iter = 15
    
    mask = torch.zeros((1,1,64,64)).cuda()
    for j in range(len(controller.word[1])):
      if j==0 or (j>=1 and controller.word[0][j]!=controller.word[0][j-1] ):
        mask += controller.adj_mask[controller.word[0][j]][3]


    loss2_threshold = {0: 0.15, 9: 0.1, 19: 0.1}

    if i < 10:
        loss3_threshold=0.95
        #loss3_threshold = 0.85
    elif 10 <= i < 20:
        loss3_threshold=0.96
        #loss3_threshold = 0.90
    else:
        loss3_threshold=0.97
        #loss3_threshold = 0.92
 
    mask = torch.zeros((1,1,64,64)).cuda()
    for j in range(len(controller.word[1])):
      if j==0 or (j>=1 and controller.word[0][j]!=controller.word[0][j-1] ):
        mask += controller.adj_mask[controller.word[0][j]][3]


    
    noise_prediction = model.unet(latents1.detach(), t, encoder_hidden_states=text_embeddings1)["sample"]
    noise_prediction = model.unet(latents2.detach(), t, encoder_hidden_states=text_embeddings2)["sample"]

    atten = controller.step_atten['up_cross'][0:3] + controller.step_atten['down_cross'][4:6]
    adj_alpha = ptp_utils1.get_word_inds(controller.prompt, controller.word[1][0], tokenizer)
    k=0
    '''for a in atten:
      print("before",i)
      a = a[:,:,adj_alpha].mean(dim=0,keepdim=True).squeeze(-1)
      a_ = a/a.max()      
      utils.save_image(a_.resize(1,16,16), "{}/before_{}_{}_{}.png".format(controller.name,controller.word[1][0],i,k), nrow=1)
      k+=1'''
    #adj_alpha = ptp_utils1.get_word_inds(controller.prompt, [1][0], tokenizer)
    # np.save(f"{controller.name}/before-{i}",torch.cat(atten).mean(0)[:,adj_alpha].detach().cpu().numpy())
    # print(atten[0].size())
    # loss1 = compute_ca_loss(atten, controller.adj_mask, ,controller.prompt,controller.name)
    # controller.get_attn_change(atten,"before")
    # print(len(atten))
    loss3 = compute_ca_loss3(atten, controller.adj_mask, controller.word, controller.prompt, controller.name)

    print("first loss3", loss3.item())
    if i < 25:

        while loss3.item() > loss3_threshold and iteration3 < loss3_max_iter:
        
            # calculate positive loss
            
            noise_prediction = model.unet(latents1, t, encoder_hidden_states=text_embeddings1)["sample"]
            noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings2)["sample"]

            atten = controller.step_atten['up_cross'][0:3] + controller.step_atten['down_cross'][4:6]
            loss3 = compute_ca_loss3(atten, controller.adj_mask, controller.word, controller.prompt, controller.name)
            
            print("loss3:", loss3.item())
            
            if controller.word_ng is not None:
              # calculate  loss
              noise_prediction = model.unet(latents1, t, encoder_hidden_states=text_embeddings1)["sample"]
              noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings3)["sample"]
             
              atten = controller.step_atten['up_cross'][0:3] + controller.step_atten['down_cross'][4:6]
              loss3_negative = compute_ca_loss3(atten, controller.adj_mask,controller.word_ng, controller.ng_prompt, controller.name)
              print("loss3_negative:", loss3_negative.item())
              
            else:
              loss3_negative = 0.0
            loss = loss3 * loss3_scale  - loss3_negative * loss3_ngscale
            
            grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents2])[0]
            sigma = sqrt((1 - model.scheduler.alphas_cumprod[t]) / model.scheduler.alphas_cumprod[t])
            latent_unmask = latents2.detach().clone() * (1 - mask)
            latent_mask = latents2 * mask
            latent_mask = latent_mask - grad_cond * sigma * mask
            latents2 = latent_unmask + latent_mask

            iteration3 += 1
            torch.cuda.empty_cache()

            with torch.no_grad():
               
                noise_prediction = model.unet(latents1, t, encoder_hidden_states=text_embeddings1)["sample"]
                noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings2)["sample"]

                atten = controller.step_atten['up_cross'][0:3] + controller.step_atten['down_cross'][4:6]
                loss3 = compute_ca_loss3(atten, controller.adj_mask,controller.word, controller.prompt,
                                         controller.name)


    if i in loss2_threshold.keys():
        with torch.no_grad():
            
            noise_prediction = model.unet(latents1, t, encoder_hidden_states=text_embeddings1)["sample"]
            noise_prediction = model.unet(latents2.detach(), t, encoder_hidden_states=text_embeddings2)["sample"]

            atten = aggregate_attention(attention_store=controller, \
                                        res=16, \
                                        from_where=("up", "down"), \
                                        is_cross=True, \
                                        select=0)
            loss2 = compute_ca_loss2(atten, controller.adj_mask, controller.word,controller.prompt, controller.name)
        print("first Loss2:", loss2.item())

        iteration2 = 0
        target_loss = max(0, 1. - loss2_threshold[i])
        while loss2 > target_loss:
           
            iteration2 += 1
            # positive
            noise_prediction = model.unet(latents1, t, encoder_hidden_states=text_embeddings1)["sample"]
            noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings2)["sample"]

            atten = aggregate_attention(attention_store=controller, \
                                        res=16, \
                                        from_where=("up", "down"), \
                                        is_cross=True, \
                                        select=0)

            loss2 = compute_ca_loss2(atten, controller.adj_mask,controller.word, controller.prompt, controller.name)
            print("Loss2:", loss2.item())
            
            if controller.word_ng is not None:
              # negative
              noise_prediction = model.unet(latents1, t, encoder_hidden_states=text_embeddings1)["sample"]
              noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings3)["sample"]
  
              atten = aggregate_attention(attention_store=controller, \
                                          res=16, \
                                          from_where=("up", "down"), \
                                          is_cross=True, \
                                          select=0)
  
              loss2_negative= compute_ca_loss2(atten, controller.adj_mask,controller.word_ng,controller.ng_prompt, controller.name)
              print("loss2_negative:", loss2_negative.item())
            else:
              loss2_negative=0.0
            
            loss = loss2 - loss2_negative * loss2_ngscale
              
              
            grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents2])[0]
            latent_unmask = latents2.detach().clone() * (1 - mask)
            latent_mask = latents2 * mask
            latent_mask = latent_mask - step_size * grad_cond * mask
            latents2 = latent_unmask + latent_mask

            with torch.no_grad():
                noise_prediction = model.unet(latents1.detach(), t, encoder_hidden_states=text_embeddings1)["sample"]
                # controller.total+=1
                noise_prediction = model.unet(latents2.detach(), t, encoder_hidden_states=text_embeddings2)["sample"]
                atten = aggregate_attention(attention_store=controller, \
                                            res=16, \
                                            from_where=("up", "down"), \
                                            is_cross=True, \
                                            select=0)
                loss2 = compute_ca_loss2(atten, controller.adj_mask,controller.word, controller.prompt, controller.name)

            if iteration2 >= loss2_max_iter:
                break
    atten = controller.step_atten['up_cross'][0:3] + controller.step_atten['down_cross'][4:6]
    k = 0
    '''for a in atten:
      a = a[:,:,adj_alpha].mean(dim=0,keepdim=True).squeeze(-1)
      print(a.size())
      a_ = a/a.max()      
      utils.save_image(a_.resize(1,16,16), "{}/after{}_{}_{}.png".format(controller.name,controller.word[1][0],i,k), nrow=1) 
      k+=1'''
    # atten = controller.step_atten['up_cross'][0:3]+controller.step_atten['down_cross'][4:6]
    # adj_alpha = ptp_utils1.get_word_inds(controller.prompt, [1][0], tokenizer)
    # np.save(f"{controller.name}/after-{i}",torch.cat(atten).mean(0)[:,adj_alpha].detach().cpu().numpy())

    return latents2.detach()









def update_latent_loss3(model, latents1, latents2,  text_embeddings1, text_embeddings2, text_embeddings3, controller, t,i):    
    loss = torch.tensor(10000)
    latents2 = latents2.requires_grad_(True)

    iteration = 0
   
    loss_scale=2.5
    loss_threshold=2.0
    loss_ngscale = 2.5
    max_iter = 15
    print("updating.....")
   
    
    mask = torch.zeros((1,1,64,64)).cuda()
    for j in range(len(controller.word[1])):
      if j==0 or (j>=1 and controller.word[0][j]!=controller.word[0][j-1] ):
        mask += controller.adj_mask[controller.word[0][j]][3]
    #print(mask)  
      
    noise_prediction = model.unet(latents1.detach(), t, encoder_hidden_states=text_embeddings1)["sample"]
    noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings2)["sample"]
    atten = controller.step_atten['up_cross'][0:3]+controller.step_atten['down_cross'][4:6]
    loss = compute_ca_loss1(atten, controller.adj_mask, controller.word,controller.prompt,controller.name)*loss_scale   
    print("first_loss1",loss.item())
    
    
    #adj_idx = ptp_utils1.get_word_inds(controller.prompt, [1][0], tokenizer)
    #print(adj_idx)
   
    #mask2 = torch.zeros(1,1,16,16).to(device)
   
    
    #for adj in controller.adj_mask:
         #mask2+=controller.adj_mask[adj][1]
    #mask2 = mask2.squeeze()
    #print(a.size())
    #print(a.max())
    #utils.save_image(a/a.max(),f"{controller.name}/atten-{i}.jpg",nrow=len(atten))
    while iteration < max_iter:
        latents2 = latents2.requires_grad_(True)
       
        #atten = controller.step_atten['up_cross'][0:3]+controller.step_atten['down_cross'][4:6]
        #a = torch.cat(atten,dim=0)[:,:,adj_idx].sum(-1).mean(0).reshape(16,16) 
        #r1 = (a * mask2).sum() / a.sum()
        #r2 = (a* (1-mask2)).sum() / a.sum()
        #np.save(f"{controller.name}/mask-{i}-{iteration}.npy",r1.detach().cpu().numpy())
        #np.save(f"{controller.name}/unmask-{i}-{iteration}.npy",r2.detach().cpu().numpy())
        #loss = compute_ca_loss1(atten, controller.adj_mask,controller.word, controller.prompt,controller.name)*loss_scale
       

        '''if controller.word_ng is not None:
          # calculate  loss
          noise_prediction = model.unet(latents1, t, encoder_hidden_states=text_embeddings1)["sample"]
          noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings3)["sample"]
          atten = controller.step_atten['up_cross'][0:3]+controller.step_atten['down_cross'][4:6]  
          loss = loss -1.0 * compute_ca_loss1(atten, controller.adj_mask,controller.word_ng, controller.ng_prompt,controller.name) * loss_ngscale  
          print("loss:",loss.item())  '''  
        
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents2] )[0]
        
        #print(grad_cond)
        
        #print("grad",grad_cond)s
        sigma = sqrt((1-model.scheduler.alphas_cumprod[t])/model.scheduler.alphas_cumprod[t])
        
        latent_unmask = latents2.detach().clone()* (1-mask)
        latent_mask = latents2.detach() * mask
        latent_mask = latent_mask - grad_cond  * mask
        
        #latent = latent - grad_cond * sigma * mask
        latents2 = latent_unmask + latent_mask
        #print(latent[0][1]==latent_copy[0][1])
        #a = (latent[0][1]==latent_copy[0][1]).float()
        #utils.save_image(a, "{}/latentMask_test.png".format(controller.name))
        #print("sigma",sigma)
        iteration += 1
       
        latents2 = latents2.requires_grad_(True)
        noise_prediction = model.unet(latents1.detach(), t, encoder_hidden_states=text_embeddings1)["sample"]
        noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings2)["sample"]
        atten = controller.step_atten['up_cross'][0:3]+controller.step_atten['down_cross'][4:6]
        loss = compute_ca_loss1(atten, controller.adj_mask,controller.word, controller.prompt,controller.name)*loss_scale
        torch.cuda.empty_cache()
    return latents2.detach()



def update_latent_loss2(model, latents1, latents2, text_embeddings1, text_embeddings2, text_embeddings3,controller, t, i):
    latents2 = latents2.requires_grad_(True)
  
    loss2 = 0
    loss = 0
    loss3 = 0
    

    iteration2 = 0
    iteration3 = 0

    loss3_ngscale= 2.5   
    loss3_scale = 2.5
 


    loss2_max_iter = 25
    loss3_max_iter = 25
    


 

    if i < 10:
        loss3_threshold=0.95
        #loss3_threshold = 0.85
    elif 10 <= i < 20:
        loss3_threshold=0.96
        #loss3_threshold = 0.90
    else:
        loss3_threshold=0.97
        #loss3_threshold = 0.92
 
    mask = torch.zeros((1,1,64,64)).cuda()
    for j in range(len(controller.word[1])):
      if j==0 or (j>=1 and controller.word[0][j]!=controller.word[0][j-1] ):
        mask += controller.adj_mask[controller.word[0][j]][3]


    
    noise_prediction = model.unet(latents1.detach(), t, encoder_hidden_states=text_embeddings1)["sample"]
    noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings2)["sample"]

    atten = controller.step_atten['up_cross'][0:3] + controller.step_atten['down_cross'][4:6]
    adj_alpha = ptp_utils1.get_word_inds(controller.prompt, controller.word[1][0], tokenizer)
    k=0
   
    loss3 = compute_ca_loss3(atten, controller.adj_mask, controller.word, controller.prompt, controller.name)

    print("first loss3", loss3.item())
    if i < 25:

        while iteration3 < loss3_max_iter:
        
            # calculate positive loss

            atten = controller.step_atten['up_cross'][0:3] + controller.step_atten['down_cross'][4:6]
            loss3 = compute_ca_loss3(atten, controller.adj_mask, controller.word, controller.prompt, controller.name)
            
            #print("loss3:", loss3.item())
            
            '''if controller.word_ng is not None:
              # calculate  loss
              noise_prediction = model.unet(latents1, t, encoder_hidden_states=text_embeddings1)["sample"]
              noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings3)["sample"]
             
              atten = controller.step_atten['up_cross'][0:3] + controller.step_atten['down_cross'][4:6]
              loss3_negative = compute_ca_loss3(atten, controller.adj_mask,controller.word_ng, controller.ng_prompt, controller.name)
              print("loss3_negative:", loss3_negative.item())
              
            else:
              loss3_negative = 0.0'''
            loss = loss3 * loss3_scale
            print(loss)
            
            grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents2])[0]
            sigma = sqrt((1 - model.scheduler.alphas_cumprod[t]) / model.scheduler.alphas_cumprod[t])
            latent_unmask = latents2.detach().clone() * (1 - mask)
            latent_mask = latents2 * mask
            latent_mask = latent_mask - grad_cond * sigma * mask
            latents2 = latent_unmask + latent_mask

            iteration3 += 1
            
               
            noise_prediction = model.unet(latents1.detach(), t, encoder_hidden_states=text_embeddings1)["sample"]
            noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings2)["sample"]

            atten = controller.step_atten['up_cross'][0:3] + controller.step_atten['down_cross'][4:6]
            loss3 = compute_ca_loss3(atten, controller.adj_mask,controller.word, controller.prompt,
                                     controller.name)
            torch.cuda.empty_cache()

    return latents2.detach()





def update_latent_loss1(model, latents1, latents2, text_embeddings1, text_embeddings2, text_embeddings3,controller, t, i):
    latents2 = latents2.requires_grad_(True)
  
    loss2 = 0
    loss = 0
    loss3 = 0
    

    iteration2 = 0
    iteration3 = 0

    loss3_ngscale= 2.5
    loss2_ngscale = 1.0
    
    loss3_scale = 2.5
    scale_range = (2., .5)
    scale_factor = 5.0
    scale_range = np.linspace(scale_range[0], scale_range[1], 50)
    step_size = scale_factor * np.sqrt(scale_range[i])
  


    loss2_max_iter = 15
    loss3_max_iter = 15
    


    loss2_threshold = {0: 0.15, 9: 0.1, 19: 0.1}

    mask = torch.zeros((1,1,64,64)).cuda()
    for j in range(len(controller.word[1])):
      if j==0 or (j>=1 and controller.word[0][j]!=controller.word[0][j-1] ):
        mask += controller.adj_mask[controller.word[0][j]][3]


    if i in loss2_threshold.keys():
        with torch.no_grad():
            
            noise_prediction = model.unet(latents1, t, encoder_hidden_states=text_embeddings1)["sample"]
            noise_prediction = model.unet(latents2.detach(), t, encoder_hidden_states=text_embeddings2)["sample"]

            atten = aggregate_attention(attention_store=controller, \
                                        res=16, \
                                        from_where=("up", "down"), \
                                        is_cross=True, \
                                        select=0)
            loss2 = compute_ca_loss2(atten, controller.adj_mask, controller.word,controller.prompt, controller.name)
        print("first Loss2:", loss2.item())

        iteration2 = 0
        target_loss = max(0, 1. - loss2_threshold[i])
        while loss2 > target_loss:
           
            iteration2 += 1
            # positive
            noise_prediction = model.unet(latents1, t, encoder_hidden_states=text_embeddings1)["sample"]
            noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings2)["sample"]

            atten = aggregate_attention(attention_store=controller, \
                                        res=16, \
                                        from_where=("up", "down"), \
                                        is_cross=True, \
                                        select=0)

            loss2 = compute_ca_loss2(atten, controller.adj_mask,controller.word, controller.prompt, controller.name)
            print("Loss2:", loss2.item())
            
            if controller.word_ng is not None:
              # negative
              noise_prediction = model.unet(latents1, t, encoder_hidden_states=text_embeddings1)["sample"]
              noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings3)["sample"]
  
              atten = aggregate_attention(attention_store=controller, \
                                          res=16, \
                                          from_where=("up", "down"), \
                                          is_cross=True, \
                                          select=0)
  
              loss2_negative= compute_ca_loss2(atten, controller.adj_mask,controller.word_ng,controller.ng_prompt, controller.name)
              print("loss2_negative:", loss2_negative.item())
            else:
              loss2_negative=0.0
            
            loss = loss2 - loss2_negative * loss2_ngscale
              
              
            grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents2])[0]
            latent_unmask = latents2.detach().clone() * (1 - mask)
            latent_mask = latents2 * mask
            latent_mask = latent_mask - step_size * grad_cond * mask
            latents2 = latent_unmask + latent_mask

            with torch.no_grad():
                noise_prediction = model.unet(latents1.detach(), t, encoder_hidden_states=text_embeddings1)["sample"]
                # controller.total+=1
                noise_prediction = model.unet(latents2.detach(), t, encoder_hidden_states=text_embeddings2)["sample"]
                atten = aggregate_attention(attention_store=controller, \
                                            res=16, \
                                            from_where=("up", "down"), \
                                            is_cross=True, \
                                            select=0)
                loss2 = compute_ca_loss2(atten, controller.adj_mask,controller.word, controller.prompt, controller.name)

            if iteration2 >= loss2_max_iter:
                break
    atten = controller.step_atten['up_cross'][0:3] + controller.step_atten['down_cross'][4:6]
  
    return latents2.detach()






 
    

@torch.no_grad()
def text2image_ldm_stable_loss1(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    uncond_embeddings=None,
    start_time=50,
    return_type='image',
    shape=None
):
    batch_size = len(prompt)
    ptp_utils1.register_attention_control(model, controller)
    ptp_utils1.register_step(model)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    text_embeddings1 = model.text_encoder(text_input.input_ids[0:1].to(model.device))[0]
    text_embeddings2 = model.text_encoder(text_input.input_ids[1:2].to(model.device))[0]
    
    if len(prompt)>2:
        text_embeddings3 = model.text_encoder(text_input.input_ids[2:3].to(model.device))[0]
    else:
       text_embeddings3=None
    max_length = text_input.input_ids.shape[-1]
    
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None
        
    if shape is not None:
        mask_idx = ptp_utils1.get_word_inds(controller.prompt, shape, tokenizer)
   
    model.scheduler.set_timesteps(num_inference_steps)
    max_index_step=10
    latent, latents = ptp_utils1.init_latent(latent, model, height, width, generator, batch_size)
    latents1 = latent.detach().clone().to(device)
    latents2 = latent.detach().clone().to(device)
    
    
    max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    
    mask = torch.zeros(1,1,64,64).cuda()
    for i in range(len(controller.word[1])):
      if i==0 or (i>=1 and controller.word[0][i]!=controller.word[0][i-1] ):
        mask += controller.adj_mask[controller.word[0][i]][3]
   
    mask =  max_pool(mask)
    #utils.save_image(mask, "{}/dilatio_mask.png".format(controller.name), nrow=1)
    
    for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):
        latent2_copy = None
        if i < 25:
            if i < 0:
                latent2_copy = latents2.detach().clone()
            controller.optimizing=True
           
            with torch.enable_grad():
              latent_optimize_new =update_latent_loss1(model, latents1, latents2,  text_embeddings1, text_embeddings2,text_embeddings3,controller, t,i)
              #latent_optimize_new =update_latent3(model, latents1, latents2,  text_embeddings1, text_embeddings2, text_embeddings3, controller, t,i)
            latents2=latent_optimize_new
            #latent2_copy = latent_optimize_new.detach().clone()
        controller.optimizing=False
        
        if uncond_embeddings_ is None:
            context1 = torch.cat([uncond_embeddings[i].expand(*text_embeddings1.shape), text_embeddings1])
            context2 = torch.cat([uncond_embeddings[i].expand(*text_embeddings2.shape), text_embeddings2])
        else:
            context1 = torch.cat([uncond_embeddings_[0:1], text_embeddings1])
            context2 = torch.cat([uncond_embeddings_[1:2], text_embeddings2])
        
        latents1,latents2 = ptp_utils1.diffusion_step11(model, controller,latents1, context1,latents2,context2, t, guidance_scale, low_resource=False,latent_before=latent2_copy)
        
        #utils.save_image(ptp_utils1.latent2image(model.vae,latents1), "{}/latents1_{}.png".format(controller.name,i), nrow=1)
        #utils.save_image(ptp_utils1.latent2image(model.vae,latents2), "{}/latents2_{}.png".format(controller.name,i), nrow=1)
        if 50-i <= 15:
            print(i)
            if shape is not None:
              atten_total = controller.attention_store['up_cross'][0:3]+controller.attention_store['down_cross'][4:6]            
              atten_total = torch.mean(torch.cat(atten_total,dim=0)[:,:,mask_idx].sum(dim=-1),dim=0)/i
              attn_mask = (atten_total/atten_total.max()).resize(16,16)
              attn_mask = attn_mask.gt(0.3).float()
              attn_mask = attn_mask[None,None,:,:]  
              mask_ca = nnf.interpolate(attn_mask, size=(64,64)).to(device)
              #mask  = mask + mask_ca
              utils.save_image(mask, "{}/avg_ca_{}.png".format(controller.name,i), nrow=1)
              
            latents2 = latents2 * mask + latents1 * (1-mask)
          
        torch.cuda.empty_cache()
        
    if return_type == 'image':
        image = ptp_utils1.latent2image(model.vae, torch.cat([latents1,latents2]))
    else:
        image = latents
    return image, latent
    
   
@torch.no_grad()
def text2image_ldm_stable_loss2(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    uncond_embeddings=None,
    start_time=50,
    return_type='image',
    shape=None
):
    batch_size = len(prompt)
    ptp_utils1.register_attention_control(model, controller)
    ptp_utils1.register_step(model)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    text_embeddings1 = model.text_encoder(text_input.input_ids[0:1].to(model.device))[0]
    text_embeddings2 = model.text_encoder(text_input.input_ids[1:2].to(model.device))[0]
    
    if len(prompt)>2:
        text_embeddings3 = model.text_encoder(text_input.input_ids[2:3].to(model.device))[0]
    else:
       text_embeddings3=None
    max_length = text_input.input_ids.shape[-1]
    
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None
        
    if shape is not None:
        mask_idx = ptp_utils1.get_word_inds(controller.prompt, shape, tokenizer)
   
    model.scheduler.set_timesteps(num_inference_steps)
    max_index_step=10
    latent, latents = ptp_utils1.init_latent(latent, model, height, width, generator, batch_size)
    latents1 = latent.detach().clone().to(device)
    latents2 = latent.detach().clone().to(device)
    
    
    max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    
    mask = torch.zeros(1,1,64,64).cuda()
    for i in range(len(controller.word[1])):
      if i==0 or (i>=1 and controller.word[0][i]!=controller.word[0][i-1] ):
        mask += controller.adj_mask[controller.word[0][i]][3]
   
    mask =  max_pool(mask)
    #utils.save_image(mask, "{}/dilatio_mask.png".format(controller.name), nrow=1)
    
    for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):
        latent2_copy = None
        if i < 25:
            if i < 0:
                latent2_copy = latents2.detach().clone()
            controller.optimizing=True
           
            with torch.enable_grad():
              latent_optimize_new =update_latent_loss2(model, latents1, latents2,  text_embeddings1, text_embeddings2,text_embeddings3,controller, t,i)
              #latent_optimize_new =update_latent3(model, latents1, latents2,  text_embeddings1, text_embeddings2, text_embeddings3, controller, t,i)
            latents2=latent_optimize_new
            #latent2_copy = latent_optimize_new.detach().clone()
        controller.optimizing=False
        
        if uncond_embeddings_ is None:
            context1 = torch.cat([uncond_embeddings[i].expand(*text_embeddings1.shape), text_embeddings1])
            context2 = torch.cat([uncond_embeddings[i].expand(*text_embeddings2.shape), text_embeddings2])
        else:
            context1 = torch.cat([uncond_embeddings_[0:1], text_embeddings1])
            context2 = torch.cat([uncond_embeddings_[1:2], text_embeddings2])
        
        latents1,latents2 = ptp_utils1.diffusion_step11(model, controller,latents1, context1,latents2,context2, t, guidance_scale, low_resource=False,latent_before=latent2_copy)
        
        #utils.save_image(ptp_utils1.latent2image(model.vae,latents1), "{}/latents1_{}.png".format(controller.name,i), nrow=1)
        #utils.save_image(ptp_utils1.latent2image(model.vae,latents2), "{}/latents2_{}.png".format(controller.name,i), nrow=1)
        if 50-i <= 15:
            print(i)
            if shape is not None:
              atten_total = controller.attention_store['up_cross'][0:3]+controller.attention_store['down_cross'][4:6]            
              atten_total = torch.mean(torch.cat(atten_total,dim=0)[:,:,mask_idx].sum(dim=-1),dim=0)/i
              attn_mask = (atten_total/atten_total.max()).resize(16,16)
              attn_mask = attn_mask.gt(0.3).float()
              attn_mask = attn_mask[None,None,:,:]  
              mask_ca = nnf.interpolate(attn_mask, size=(64,64)).to(device)
              #mask  = mask + mask_ca
              utils.save_image(mask, "{}/avg_ca_{}.png".format(controller.name,i), nrow=1)
              
            latents2 = latents2 * mask + latents1 * (1-mask)
          
        torch.cuda.empty_cache()
        
    if return_type == 'image':
        image = ptp_utils1.latent2image(model.vae, torch.cat([latents1,latents2]))
    else:
        image = latents
    return image, latent

@torch.no_grad()
def text2image_ldm_stable_loss3(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    uncond_embeddings=None,
    start_time=50,
    return_type='image',
    shape=None
):
    batch_size = len(prompt)
    ptp_utils1.register_attention_control(model, controller)
    ptp_utils1.register_step(model)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    text_embeddings1 = model.text_encoder(text_input.input_ids[0:1].to(model.device))[0]
    text_embeddings2 = model.text_encoder(text_input.input_ids[1:2].to(model.device))[0]
    
    if len(prompt)>2:
        text_embeddings3 = model.text_encoder(text_input.input_ids[2:3].to(model.device))[0]
    else:
       text_embeddings3=None
    max_length = text_input.input_ids.shape[-1]
    
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None
        
    if shape is not None:
        mask_idx = ptp_utils1.get_word_inds(controller.prompt, shape, tokenizer)
   
    model.scheduler.set_timesteps(num_inference_steps)
    max_index_step=10
    latent, latents = ptp_utils1.init_latent(latent, model, height, width, generator, batch_size)
    latents1 = latent.detach().clone().to(device)
    latents2 = latent.detach().clone().to(device)
    
    
    max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    
    mask = torch.zeros(1,1,64,64).cuda()
    for i in range(len(controller.word[1])):
      if i==0 or (i>=1 and controller.word[0][i]!=controller.word[0][i-1] ):
        mask += controller.adj_mask[controller.word[0][i]][3]
   
    mask =  max_pool(mask)
    #utils.save_image(mask, "{}/dilatio_mask.png".format(controller.name), nrow=1)
    
    for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):
        latent2_copy = None
        if i < 25:
            if i < 0:
                latent2_copy = latents2.detach().clone()
            controller.optimizing=True
           
            with torch.enable_grad():
              latent_optimize_new =update_latent_loss3(model, latents1, latents2,  text_embeddings1, text_embeddings2,text_embeddings3,controller, t,i)
              #latent_optimize_new =update_latent3(model, latents1, latents2,  text_embeddings1, text_embeddings2, text_embeddings3, controller, t,i)
            latents2=latent_optimize_new
            #latent2_copy = latent_optimize_new.detach().clone()
        controller.optimizing=False
        
        if uncond_embeddings_ is None:
            context1 = torch.cat([uncond_embeddings[i].expand(*text_embeddings1.shape), text_embeddings1])
            context2 = torch.cat([uncond_embeddings[i].expand(*text_embeddings2.shape), text_embeddings2])
        else:
            context1 = torch.cat([uncond_embeddings_[0:1], text_embeddings1])
            context2 = torch.cat([uncond_embeddings_[1:2], text_embeddings2])
        
        latents1,latents2 = ptp_utils1.diffusion_step11(model, controller,latents1, context1,latents2,context2, t, guidance_scale, low_resource=False,latent_before=latent2_copy)
        
        #utils.save_image(ptp_utils1.latent2image(model.vae,latents1), "{}/latents1_{}.png".format(controller.name,i), nrow=1)
        #utils.save_image(ptp_utils1.latent2image(model.vae,latents2), "{}/latents2_{}.png".format(controller.name,i), nrow=1)
        if 50-i <= 15:
            print(i)
            if shape is not None:
              atten_total = controller.attention_store['up_cross'][0:3]+controller.attention_store['down_cross'][4:6]            
              atten_total = torch.mean(torch.cat(atten_total,dim=0)[:,:,mask_idx].sum(dim=-1),dim=0)/i
              attn_mask = (atten_total/atten_total.max()).resize(16,16)
              attn_mask = attn_mask.gt(0.3).float()
              attn_mask = attn_mask[None,None,:,:]  
              mask_ca = nnf.interpolate(attn_mask, size=(64,64)).to(device)
              #mask  = mask + mask_ca
              utils.save_image(mask, "{}/avg_ca_{}.png".format(controller.name,i), nrow=1)
              
            latents2 = latents2 * mask + latents1 * (1-mask)
          
        torch.cuda.empty_cache()
        
    if return_type == 'image':
        image = ptp_utils1.latent2image(model.vae, torch.cat([latents1,latents2]))
    else:
        image = latents
    return image, latent
    
    
@torch.no_grad()
def text2image_ldm_stable_loss12(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    uncond_embeddings=None,
    start_time=50,
    return_type='image',
    shape=None
):
    batch_size = len(prompt)
    ptp_utils1.register_attention_control(model, controller)
    ptp_utils1.register_step(model)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    text_embeddings1 = model.text_encoder(text_input.input_ids[0:1].to(model.device))[0]
    text_embeddings2 = model.text_encoder(text_input.input_ids[1:2].to(model.device))[0]
    
    if len(prompt)>2:
        text_embeddings3 = model.text_encoder(text_input.input_ids[2:3].to(model.device))[0]
    else:
       text_embeddings3=None
    max_length = text_input.input_ids.shape[-1]
    
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None
        
    if shape is not None:
        mask_idx = ptp_utils1.get_word_inds(controller.prompt, shape, tokenizer)
   
    model.scheduler.set_timesteps(num_inference_steps)
    max_index_step=10
    latent, latents = ptp_utils1.init_latent(latent, model, height, width, generator, batch_size)
    latents1 = latent.detach().clone().to(device)
    latents2 = latent.detach().clone().to(device)
    
    
    max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    
    mask = torch.zeros(1,1,64,64).cuda()
    for i in range(len(controller.word[1])):
      if i==0 or (i>=1 and controller.word[0][i]!=controller.word[0][i-1] ):
        mask += controller.adj_mask[controller.word[0][i]][3]
   
    mask =  max_pool(mask)
    #utils.save_image(mask, "{}/dilatio_mask.png".format(controller.name), nrow=1)
    
    for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):
        latent2_copy = None
        if i < 25:
            if i < 0:
                latent2_copy = latents2.detach().clone()
            controller.optimizing=True
           
            with torch.enable_grad():
              latent_optimize_new =update_latent2_loss12(model, latents1, latents2,  text_embeddings1, text_embeddings2,text_embeddings3,controller, t,i)
              #latent_optimize_new =update_latent3(model, latents1, latents2,  text_embeddings1, text_embeddings2, text_embeddings3, controller, t,i)
            latents2=latent_optimize_new
            #latent2_copy = latent_optimize_new.detach().clone()
        controller.optimizing=False
        
        if uncond_embeddings_ is None:
            context1 = torch.cat([uncond_embeddings[i].expand(*text_embeddings1.shape), text_embeddings1])
            context2 = torch.cat([uncond_embeddings[i].expand(*text_embeddings2.shape), text_embeddings2])
        else:
            context1 = torch.cat([uncond_embeddings_[0:1], text_embeddings1])
            context2 = torch.cat([uncond_embeddings_[1:2], text_embeddings2])
        
        latents1,latents2 = ptp_utils1.diffusion_step11(model, controller,latents1, context1,latents2,context2, t, guidance_scale, low_resource=False,latent_before=latent2_copy)
        
        #utils.save_image(ptp_utils1.latent2image(model.vae,latents1), "{}/latents1_{}.png".format(controller.name,i), nrow=1)
        #utils.save_image(ptp_utils1.latent2image(model.vae,latents2), "{}/latents2_{}.png".format(controller.name,i), nrow=1)
        if 50-i <= 15:
            print(i)
              
            latents2 = latents2 * mask + latents1 * (1-mask)
          
        torch.cuda.empty_cache()
        
    if return_type == 'image':
        image = ptp_utils1.latent2image(model.vae, torch.cat([latents1,latents2]))
    else:
        image = latents
    return image, latent
    
   
@torch.no_grad()
def text2image_ldm_stable_loss13(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    uncond_embeddings=None,
    start_time=50,
    return_type='image',
    shape=None
):
    batch_size = len(prompt)
    ptp_utils1.register_attention_control(model, controller)
    ptp_utils1.register_step(model)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    text_embeddings1 = model.text_encoder(text_input.input_ids[0:1].to(model.device))[0]
    text_embeddings2 = model.text_encoder(text_input.input_ids[1:2].to(model.device))[0]
    
    if len(prompt)>2:
        text_embeddings3 = model.text_encoder(text_input.input_ids[2:3].to(model.device))[0]
    else:
       text_embeddings3=None
    max_length = text_input.input_ids.shape[-1]
    
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None
        
    if shape is not None:
        mask_idx = ptp_utils1.get_word_inds(controller.prompt, shape, tokenizer)
   
    model.scheduler.set_timesteps(num_inference_steps)
    max_index_step=10
    latent, latents = ptp_utils1.init_latent(latent, model, height, width, generator, batch_size)
    latents1 = latent.detach().clone().to(device)
    latents2 = latent.detach().clone().to(device)
    
    
    max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    
    mask = torch.zeros(1,1,64,64).cuda()
    for i in range(len(controller.word[1])):
      if i==0 or (i>=1 and controller.word[0][i]!=controller.word[0][i-1] ):
        mask += controller.adj_mask[controller.word[0][i]][3]
   
    mask =  max_pool(mask)
    #utils.save_image(mask, "{}/dilatio_mask.png".format(controller.name), nrow=1)
    
    for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):
        latent2_copy = None
        if i < 25:
            if i < 0:
                latent2_copy = latents2.detach().clone()
            controller.optimizing=True
           
            with torch.enable_grad():
              latent_optimize_new =update_latent2_loss13(model, latents1, latents2,  text_embeddings1, text_embeddings2,text_embeddings3,controller, t,i)
              #latent_optimize_new =update_latent3(model, latents1, latents2,  text_embeddings1, text_embeddings2, text_embeddings3, controller, t,i)
            latents2=latent_optimize_new
            #latent2_copy = latent_optimize_new.detach().clone()
        controller.optimizing=False
        
        if uncond_embeddings_ is None:
            context1 = torch.cat([uncond_embeddings[i].expand(*text_embeddings1.shape), text_embeddings1])
            context2 = torch.cat([uncond_embeddings[i].expand(*text_embeddings2.shape), text_embeddings2])
        else:
            context1 = torch.cat([uncond_embeddings_[0:1], text_embeddings1])
            context2 = torch.cat([uncond_embeddings_[1:2], text_embeddings2])
        
        latents1,latents2 = ptp_utils1.diffusion_step11(model, controller,latents1, context1,latents2,context2, t, guidance_scale, low_resource=False,latent_before=latent2_copy)
        
        #utils.save_image(ptp_utils1.latent2image(model.vae,latents1), "{}/latents1_{}.png".format(controller.name,i), nrow=1)
        #utils.save_image(ptp_utils1.latent2image(model.vae,latents2), "{}/latents2_{}.png".format(controller.name,i), nrow=1)
        if 50-i <= 15:
            print(i)
            if shape is not None:
              atten_total = controller.attention_store['up_cross'][0:3]+controller.attention_store['down_cross'][4:6]            
              atten_total = torch.mean(torch.cat(atten_total,dim=0)[:,:,mask_idx].sum(dim=-1),dim=0)/i
              attn_mask = (atten_total/atten_total.max()).resize(16,16)
              attn_mask = attn_mask.gt(0.3).float()
              attn_mask = attn_mask[None,None,:,:]  
              mask_ca = nnf.interpolate(attn_mask, size=(64,64)).to(device)
              #mask  = mask + mask_ca
              utils.save_image(mask, "{}/avg_ca_{}.png".format(controller.name,i), nrow=1)
              
            latents2 = latents2 * mask + latents1 * (1-mask)
          
        torch.cuda.empty_cache()
        
    if return_type == 'image':
        image = ptp_utils1.latent2image(model.vae, torch.cat([latents1,latents2]))
    else:
        image = latents
    return image, latent
    
    
@torch.no_grad()
def text2image_ldm_stable_loss23(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    uncond_embeddings=None,
    start_time=50,
    return_type='image',
    shape=None
):
    batch_size = len(prompt)
    ptp_utils1.register_attention_control(model, controller)
    ptp_utils1.register_step(model)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    text_embeddings1 = model.text_encoder(text_input.input_ids[0:1].to(model.device))[0]
    text_embeddings2 = model.text_encoder(text_input.input_ids[1:2].to(model.device))[0]
    
    if len(prompt)>2:
        text_embeddings3 = model.text_encoder(text_input.input_ids[2:3].to(model.device))[0]
    else:
       text_embeddings3=None
    max_length = text_input.input_ids.shape[-1]
    
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None
        
    if shape is not None:
        mask_idx = ptp_utils1.get_word_inds(controller.prompt, shape, tokenizer)
   
    model.scheduler.set_timesteps(num_inference_steps)
    max_index_step=10
    latent, latents = ptp_utils1.init_latent(latent, model, height, width, generator, batch_size)
    latents1 = latent.detach().clone().to(device)
    latents2 = latent.detach().clone().to(device)
    
    
    max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    
    mask = torch.zeros(1,1,64,64).cuda()
    for i in range(len(controller.word[1])):
      if i==0 or (i>=1 and controller.word[0][i]!=controller.word[0][i-1] ):
        mask += controller.adj_mask[controller.word[0][i]][3]
   
    mask =  max_pool(mask)
    #utils.save_image(mask, "{}/dilatio_mask.png".format(controller.name), nrow=1)
    
    for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):
        latent2_copy = None
        if i < 25:
            if i < 0:
                latent2_copy = latents2.detach().clone()
            controller.optimizing=True
           
            with torch.enable_grad():
              latent_optimize_new =update_latent2_loss23(model, latents1, latents2,  text_embeddings1, text_embeddings2,text_embeddings3,controller, t,i)
              #latent_optimize_new =update_latent3(model, latents1, latents2,  text_embeddings1, text_embeddings2, text_embeddings3, controller, t,i)
            latents2=latent_optimize_new
            #latent2_copy = latent_optimize_new.detach().clone()
        controller.optimizing=False
        
        if uncond_embeddings_ is None:
            context1 = torch.cat([uncond_embeddings[i].expand(*text_embeddings1.shape), text_embeddings1])
            context2 = torch.cat([uncond_embeddings[i].expand(*text_embeddings2.shape), text_embeddings2])
        else:
            context1 = torch.cat([uncond_embeddings_[0:1], text_embeddings1])
            context2 = torch.cat([uncond_embeddings_[1:2], text_embeddings2])
        
        latents1,latents2 = ptp_utils1.diffusion_step11(model, controller,latents1, context1,latents2,context2, t, guidance_scale, low_resource=False,latent_before=latent2_copy)
        
        #utils.save_image(ptp_utils1.latent2image(model.vae,latents1), "{}/latents1_{}.png".format(controller.name,i), nrow=1)
        #utils.save_image(ptp_utils1.latent2image(model.vae,latents2), "{}/latents2_{}.png".format(controller.name,i), nrow=1)
        if 50-i <= 15:
            print(i)
            if shape is not None:
              atten_total = controller.attention_store['up_cross'][0:3]+controller.attention_store['down_cross'][4:6]            
              atten_total = torch.mean(torch.cat(atten_total,dim=0)[:,:,mask_idx].sum(dim=-1),dim=0)/i
              attn_mask = (atten_total/atten_total.max()).resize(16,16)
              attn_mask = attn_mask.gt(0.3).float()
              attn_mask = attn_mask[None,None,:,:]  
              mask_ca = nnf.interpolate(attn_mask, size=(64,64)).to(device)
              #mask  = mask + mask_ca
              utils.save_image(mask, "{}/avg_ca_{}.png".format(controller.name,i), nrow=1)
              
            latents2 = latents2 * mask + latents1 * (1-mask)
          
        torch.cuda.empty_cache()
        
    if return_type == 'image':
        image = ptp_utils1.latent2image(model.vae, torch.cat([latents1,latents2]))
    else:
        image = latents
    return image, latent
    
    
@torch.no_grad()
def text2image_ldm_stable_loss123(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    uncond_embeddings=None,
    start_time=50,
    return_type='image',
    shape=None
):
    batch_size = len(prompt)
    ptp_utils1.register_attention_control(model, controller)
    ptp_utils1.register_step(model)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    text_embeddings1 = model.text_encoder(text_input.input_ids[0:1].to(model.device))[0]
    text_embeddings2 = model.text_encoder(text_input.input_ids[1:2].to(model.device))[0]
    
    if len(prompt)>2:
        text_embeddings3 = model.text_encoder(text_input.input_ids[2:3].to(model.device))[0]
    else:
       text_embeddings3=None
    max_length = text_input.input_ids.shape[-1]
    
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None
        
    if shape is not None:
        mask_idx = ptp_utils1.get_word_inds(controller.prompt, shape, tokenizer)
   
    model.scheduler.set_timesteps(num_inference_steps)
    max_index_step=10
    latent, latents = ptp_utils1.init_latent(latent, model, height, width, generator, batch_size)
    latents1 = latent.detach().clone().to(device)
    latents2 = latent.detach().clone().to(device)
    
    
    max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    
    mask = torch.zeros(1,1,64,64).cuda()
    for i in range(len(controller.word[1])):
      if i==0 or (i>=1 and controller.word[0][i]!=controller.word[0][i-1] ):
        mask += controller.adj_mask[controller.word[0][i]][3]
   
    mask =  max_pool(mask)
    #utils.save_image(mask, "{}/dilatio_mask.png".format(controller.name), nrow=1)
    
    for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):
        latent2_copy = None
        if i < 25:
            if i < 0:
                latent2_copy = latents2.detach().clone()
            controller.optimizing=True
           
            with torch.enable_grad():
              latent_optimize_new =update_latent4(model, latents1, latents2,  text_embeddings1, text_embeddings2,text_embeddings3,controller, t,i)
              #latent_optimize_new =update_latent3(model, latents1, latents2,  text_embeddings1, text_embeddings2, text_embeddings3, controller, t,i)
            latents2=latent_optimize_new
            #latent2_copy = latent_optimize_new.detach().clone()
        controller.optimizing=False
        
        if uncond_embeddings_ is None:
            context1 = torch.cat([uncond_embeddings[i].expand(*text_embeddings1.shape), text_embeddings1])
            context2 = torch.cat([uncond_embeddings[i].expand(*text_embeddings2.shape), text_embeddings2])
        else:
            context1 = torch.cat([uncond_embeddings_[0:1], text_embeddings1])
            context2 = torch.cat([uncond_embeddings_[1:2], text_embeddings2])
        
        latents1,latents2 = ptp_utils1.diffusion_step11(model, controller,latents1, context1,latents2,context2, t, guidance_scale, low_resource=False,latent_before=latent2_copy)
        
        #utils.save_image(ptp_utils1.latent2image(model.vae,latents1), "{}/latents1_{}.png".format(controller.name,i), nrow=1)
        #utils.save_image(ptp_utils1.latent2image(model.vae,latents2), "{}/latents2_{}.png".format(controller.name,i), nrow=1)
        if 50-i <= 15:
            print(i)
            if shape is not None:
              atten_total = controller.attention_store['up_cross'][0:3]+controller.attention_store['down_cross'][4:6]            
              atten_total = torch.mean(torch.cat(atten_total,dim=0)[:,:,mask_idx].sum(dim=-1),dim=0)/i
              attn_mask = (atten_total/atten_total.max()).resize(16,16)
              attn_mask = attn_mask.gt(0.3).float()
              attn_mask = attn_mask[None,None,:,:]  
              mask_ca = nnf.interpolate(attn_mask, size=(64,64)).to(device)
              #mask  = mask + mask_ca
              utils.save_image(mask, "{}/avg_ca_{}.png".format(controller.name,i), nrow=1)
              
            latents2 = latents2 * mask + latents1 * (1-mask)
          
        torch.cuda.empty_cache()
        
    if return_type == 'image':
        image = ptp_utils1.latent2image(model.vae, torch.cat([latents1,latents2]))
    else:
        image = latents
    return image, latent
 
    
@torch.no_grad()
def text2image_ldm_stable_lossadd_123(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    uncond_embeddings=None,
    start_time=50,
    return_type='image',
    shape=None
):
    batch_size = len(prompt)
    ptp_utils1.register_attention_control(model, controller)
    ptp_utils1.register_step(model)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    text_embeddings1 = model.text_encoder(text_input.input_ids[0:1].to(model.device))[0]
    text_embeddings2 = model.text_encoder(text_input.input_ids[1:2].to(model.device))[0]
    
    if len(prompt)>2:
        text_embeddings3 = model.text_encoder(text_input.input_ids[2:3].to(model.device))[0]
    else:
       text_embeddings3=None
    max_length = text_input.input_ids.shape[-1]
    
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None
        
    if shape is not None:
        mask_idx = ptp_utils1.get_word_inds(controller.prompt, shape, tokenizer)
   
    model.scheduler.set_timesteps(num_inference_steps)
    max_index_step=10
    latent, latents = ptp_utils1.init_latent(latent, model, height, width, generator, batch_size)
    latents1 = latent.detach().clone().to(device)
    latents2 = latent.detach().clone().to(device)
    
    
    max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    
    mask = torch.zeros(1,1,64,64).cuda()
    for i in range(len(controller.word[1])):
      if i==0 or (i>=1 and controller.word[0][i]!=controller.word[0][i-1] ):
        mask += controller.adj_mask[controller.word[0][i]][3]
   
    mask =  max_pool(mask)
    utils.save_image(mask, "{}/dilatio_mask.png".format(controller.name), nrow=1)
    
    for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):
        latent2_copy = None
        if i < 25:
            if i < 0:
                latent2_copy = latents2.detach().clone()
            controller.optimizing=True
           
            with torch.enable_grad():
              latent_optimize_new =update_latent_lossadd123(model, latents1, latents2,  text_embeddings1, text_embeddings2,text_embeddings3,controller, t,i)
              #latent_optimize_new =update_latent3(model, latents1, latents2,  text_embeddings1, text_embeddings2, text_embeddings3, controller, t,i)
            latents2=latent_optimize_new
            #latent2_copy = latent_optimize_new.detach().clone()
        controller.optimizing=False
        
        if uncond_embeddings_ is None:
            context1 = torch.cat([uncond_embeddings[i].expand(*text_embeddings1.shape), text_embeddings1])
            context2 = torch.cat([uncond_embeddings[i].expand(*text_embeddings2.shape), text_embeddings2])
        else:
            context1 = torch.cat([uncond_embeddings_[0:1], text_embeddings1])
            context2 = torch.cat([uncond_embeddings_[1:2], text_embeddings2])
        
        latents1,latents2 = ptp_utils1.diffusion_step11(model, controller,latents1, context1,latents2,context2, t, guidance_scale, low_resource=False,latent_before=latent2_copy)
        
        #utils.save_image(ptp_utils1.latent2image(model.vae,latents1), "{}/latents1_{}.png".format(controller.name,i), nrow=1)
        #utils.save_image(ptp_utils1.latent2image(model.vae,latents2), "{}/latents2_{}.png".format(controller.name,i), nrow=1)
        if 50-i <= 15:
            #print(i)
            if shape is not None:
              atten_total = controller.attention_store['up_cross'][0:3]+controller.attention_store['down_cross'][4:6]            
              atten_total = torch.mean(torch.cat(atten_total,dim=0)[:,:,mask_idx].sum(dim=-1),dim=0)/i
              attn_mask = (atten_total/atten_total.max()).resize(16,16)
              attn_mask = attn_mask.gt(0.3).float()
              attn_mask = attn_mask[None,None,:,:]  
              mask_ca = nnf.interpolate(attn_mask, size=(64,64)).to(device)
              #mask  = mask + mask_ca
              utils.save_image(mask, "{}/avg_ca_{}.png".format(controller.name,i), nrow=1)
              
            latents2 = latents2 * mask + latents1 * (1-mask)
          
        torch.cuda.empty_cache()
        
    if return_type == 'image':
        image = ptp_utils1.latent2image(model.vae, torch.cat([latents1,latents2]))
    else:
        image = latents
    return image, latent
    
@torch.no_grad()
def text2image_ldm_stable_lossadd_123_wo_v(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    uncond_embeddings=None,
    start_time=50,
    return_type='image',
    shape=None
):
    batch_size = len(prompt)
    ptp_utils1.register_attention_control(model, controller)
    ptp_utils1.register_step(model)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    text_embeddings1 = model.text_encoder(text_input.input_ids[0:1].to(model.device))[0]
    text_embeddings2 = model.text_encoder(text_input.input_ids[1:2].to(model.device))[0]
    
    if len(prompt)>2:
        text_embeddings3 = model.text_encoder(text_input.input_ids[2:3].to(model.device))[0]
    else:
       text_embeddings3=None
    max_length = text_input.input_ids.shape[-1]
    
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None
        
    if shape is not None:
        mask_idx = ptp_utils1.get_word_inds(controller.prompt, shape, tokenizer)
   
    model.scheduler.set_timesteps(num_inference_steps)
    max_index_step=10
    latent, latents = ptp_utils1.init_latent(latent, model, height, width, generator, batch_size)
    latents1 = latent.detach().clone().to(device)
    latents2 = latent.detach().clone().to(device)
    
    
    max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    
    mask = torch.zeros(1,1,64,64).cuda()
    for i in range(len(controller.word[1])):
      if i==0 or (i>=1 and controller.word[0][i]!=controller.word[0][i-1] ):
        mask += controller.adj_mask[controller.word[0][i]][3]
   
    mask =  max_pool(mask)
    #utils.save_image(mask, "{}/dilatio_mask.png".format(controller.name), nrow=1)
    
    for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):
        latent2_copy = None
        if i < 25:
            if i < 0:
                latent2_copy = latents2.detach().clone()
            controller.optimizing=True
           
            with torch.enable_grad():
              latent_optimize_new =update_latent_lossadd123_wo_v(model, latents1, latents2,  text_embeddings1, text_embeddings2,text_embeddings3,controller, t,i)
              #latent_optimize_new =update_latent3(model, latents1, latents2,  text_embeddings1, text_embeddings2, text_embeddings3, controller, t,i)
            latents2=latent_optimize_new
            #latent2_copy = latent_optimize_new.detach().clone()
        controller.optimizing=False
        
        if uncond_embeddings_ is None:
            context1 = torch.cat([uncond_embeddings[i].expand(*text_embeddings1.shape), text_embeddings1])
            context2 = torch.cat([uncond_embeddings[i].expand(*text_embeddings2.shape), text_embeddings2])
        else:
            context1 = torch.cat([uncond_embeddings_[0:1], text_embeddings1])
            context2 = torch.cat([uncond_embeddings_[1:2], text_embeddings2])
        
        latents1,latents2 = ptp_utils1.diffusion_step11(model, controller,latents1, context1,latents2,context2, t, guidance_scale, low_resource=False,latent_before=latent2_copy)
        
        #utils.save_image(ptp_utils1.latent2image(model.vae,latents1), "{}/latents1_{}.png".format(controller.name,i), nrow=1)
        #utils.save_image(ptp_utils1.latent2image(model.vae,latents2), "{}/latents2_{}.png".format(controller.name,i), nrow=1)
        
        if 50-i <= 15:
            print(i)
            '''if shape is not None:
            atten_total = controller.attention_store['up_cross'][0:3]+controller.attention_store['down_cross'][4:6]            
            atten_total = torch.mean(torch.cat(atten_total,dim=0)[:,:,mask_idx].sum(dim=-1),dim=0)/i
            attn_mask = (atten_total/atten_total.max()).resize(16,16)
            attn_mask = attn_mask.gt(0.3).float()
            attn_mask = attn_mask[None,None,:,:]  
            mask_ca = nnf.interpolate(attn_mask, size=(64,64)).to(device)
            mask  = mask + mask_ca
            utils.save_image(mask, "{}/avg_ca_{}.png".format(controller.name,i), nrow=1)'''
            latents2 = latents2 * mask + latents1 * (1-mask)
            utils.save_image(mask, "{}/mask{}.png".format(controller.name,i), nrow=1)  
        torch.cuda.empty_cache()
        
    if return_type == 'image':
        image = ptp_utils1.latent2image(model.vae, torch.cat([latents1,latents2]))
    else:
        image = latents
    return image, latent
        
    
@torch.no_grad()
def text2image_ldm_stable_lossadd_123_(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    uncond_embeddings=None,
    start_time=50,
    return_type='image',
    shape=None
):
    batch_size = len(prompt)
    ptp_utils1.register_attention_control(model, controller)
    ptp_utils1.register_step(model)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    text_embeddings1 = model.text_encoder(text_input.input_ids[0:1].to(model.device))[0]
    text_embeddings2 = model.text_encoder(text_input.input_ids[1:2].to(model.device))[0]
    
    if len(prompt)>2:
        text_embeddings3 = model.text_encoder(text_input.input_ids[2:3].to(model.device))[0]
    else:
       text_embeddings3=None
    max_length = text_input.input_ids.shape[-1]
    
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None
        
    if shape is not None:
        mask_idx = ptp_utils1.get_word_inds(controller.prompt, shape, tokenizer)
   
    model.scheduler.set_timesteps(num_inference_steps)
    max_index_step=10
    latent, latents = ptp_utils1.init_latent(latent, model, height, width, generator, batch_size)
    latents1 = latent.detach().clone().to(device)
    latents2 = latent.detach().clone().to(device)
    
    
    max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    
    mask = torch.zeros(1,1,64,64).cuda()
    for i in range(len(controller.word[1])):
      if i==0 or (i>=1 and controller.word[0][i]!=controller.word[0][i-1] ):
        mask += controller.adj_mask[controller.word[0][i]][3]
   
    mask =  max_pool(mask)
    #utils.save_image(mask, "{}/dilatio_mask.png".format(controller.name), nrow=1)
    
    for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):
        latent2_copy = None
        if i < 25:
            if i < 0:
                latent2_copy = latents2.detach().clone()
            controller.optimizing=True
           
            with torch.enable_grad():
              latent_optimize_new =update_latent_lossadd123_(model, latents1, latents2,  text_embeddings1, text_embeddings2,text_embeddings3,controller, t,i)
              #latent_optimize_new =update_latent3(model, latents1, latents2,  text_embeddings1, text_embeddings2, text_embeddings3, controller, t,i)
            latents2=latent_optimize_new
            #latent2_copy = latent_optimize_new.detach().clone()
        controller.optimizing=False
        
        if uncond_embeddings_ is None:
            context1 = torch.cat([uncond_embeddings[i].expand(*text_embeddings1.shape), text_embeddings1])
            context2 = torch.cat([uncond_embeddings[i].expand(*text_embeddings2.shape), text_embeddings2])
        else:
            context1 = torch.cat([uncond_embeddings_[0:1], text_embeddings1])
            context2 = torch.cat([uncond_embeddings_[1:2], text_embeddings2])
        
        latents1,latents2 = ptp_utils1.diffusion_step11(model, controller,latents1, context1,latents2,context2, t, guidance_scale, low_resource=False,latent_before=latent2_copy)
        
        #utils.save_image(ptp_utils1.latent2image(model.vae,latents1), "{}/latents1_{}.png".format(controller.name,i), nrow=1)
        #utils.save_image(ptp_utils1.latent2image(model.vae,latents2), "{}/latents2_{}.png".format(controller.name,i), nrow=1)
        if 50-i <= 15:
            
            
              
            latents2 = latents2 * mask + latents1 * (1-mask)
          
        torch.cuda.empty_cache()
        
    if return_type == 'image':
        image = ptp_utils1.latent2image(model.vae, torch.cat([latents1,latents2]))
    else:
        image = latents
    return image, latent