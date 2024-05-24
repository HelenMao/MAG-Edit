from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm import tqdm
import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn.functional as nnf
import numpy as np
import abc
import ptp_utils
import seq_aligner
from torch.optim.adam import Adam
from PIL import Image
from torchvision import utils 
import os
from math import sqrt
import torchvision.transforms as transforms
from einops import rearrange, repeat
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
MY_TOKEN = ''
LOW_RESOURCE = False 
NUM_DDIM_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
ldm_stable = StableDiffusionPipeline.from_pretrained("/data/home/models/stable-diffusion-v1-4/", scheduler=scheduler).to(device)
try:
    ldm_stable.disable_xformers_memory_efficient_attention()
except AttributeError:
    print("Attribute disable_xformers_memory_efficient_attention() is missing")
tokenizer = ldm_stable.tokenizer

print(ldm_stable.scheduler.config.steps_offset)


class AttentionBase:
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def after_step(self):
        pass

    def __call__(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads,scale, **kwargs):
        out = self.forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            # after step
            self.after_step()
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)
        return out

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

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
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1

        if substruct_words is not None:
            substruct_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
            for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
                if type(words_) is str:
                    words_ = [words_]
                for word in words_:
                    ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
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
             

              attn = self.forward(attn, is_cross, place_in_unet)

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

      self.step_store[key].append(attn)
      return attn
        
    def save_optimize_attn(self, attn, is_cross,place_in_unet):
       
        if is_cross:
            key = f"{place_in_unet}_{'cross'}"
            self.step_atten[key].append(attn)
    

            

    def between_steps(self):
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
        self.attention_store = {}
        self.step_atten = {"down_cross": [],  "up_cross": [],  "mid_cross": []}
        self.name=name
        if self.name:
            os.makedirs(name,exist_ok=True)
        print(self.name)
        self.optimizing=False
        self.att_range=[]



        
class AttentionControlEdit(AttentionStore, abc.ABC):
    
    def step_callback(self, x_t,t=0):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store,t)
        return x_t
        
    def replace_self_attention(self, attn ,place_in_unet):
    #把reference的self attn存起来在特定阶段替换
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
                attn = attn_repalce_new
            else:
                attn = self.replace_self_attention(attn, place_in_unet)
           
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
            
            
            self.count_attn[key] +=1        
        
        return attn
 
            
    
    def mask_n(self):
        #cross16 = torch.Tensor(cluster).to(device)
        adj_mask={}
        transform = transforms.ToTensor()
        for n  in self.word[0]:

            img = Image.open(n).convert('L')
            #n_alpha = self.get_n_alpha(n)
            maskk = transform(img)
            maskk = maskk.unsqueeze(0)            
            
            mask = nnf.interpolate(maskk, size=(8,8)).to(device)
            mask = mask!=0   
            adj_mask[n] = [mask]


            mask = nnf.interpolate(maskk, size=(16,16)).to(device)   
            mask = mask!=0   
            adj_mask[n].append(mask)

            
            mask = nnf.interpolate(maskk, size=(32,32)).to(device)   
            mask = mask!=0   
            adj_mask[n].append(mask)

            
            mask = nnf.interpolate(maskk, size=(64,64))
            mask = mask!=0
            adj_mask[n].append(mask.to(device))

        return adj_mask


    
    
    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend]
                ,name=None,word=None,adj_mask=None,word_ng=None):
        
        super(AttentionControlEdit, self).__init__(name)
        self.batch_size = 1
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts[:2], num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend
        self.word = word
        self.word_ng=word_ng
        self.prompt = prompts[1]
        self.ng_prompt = prompts[2] if word_ng is not None else None
        self.adj_mask =  self.mask_n()

        
        
class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, att_replace,place_in_unet):
        key = f"{place_in_unet}_{'cross'}"
        attn_base = self.step_store[key][self.count_attn[key]]
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
      
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None,name=None,word=None,adj_mask=None,word_ng=None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend,name=name,word=word,adj_mask=adj_mask,word_ng=word_ng)
        self.mapper = seq_aligner.get_replacement_mapper(prompts[:2], tokenizer).to(device)
        #print(self.mapper)

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

        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
                local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                  Tuple[float, ...]]):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(1, 77)
    
    for word, val in zip(word_select, values):
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = val
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
    return out


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
        image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    ptp_utils.view_images(np.stack(images, axis=0))
    

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
    ptp_utils.view_images(np.concatenate(images, axis=1))
    
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
        ptp_utils.register_attention_control(self.model, None)
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
        




def compute_ca_TR(attn_maps, masks, word,prompt):
    loss = 0.0
    
    #print(word,prompt)
    for obj_idx in range(len(word[1])):
        if  word[1][obj_idx] not in prompt:
            continue  
        adj_idx = ptp_utils.get_word_inds(prompt, word[1][obj_idx], tokenizer)
        
        
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



def compute_ca_SR(attn_maps, masks, word,prompt,name):
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
             adj_idx = ptp_utils.get_word_inds(prompt, word[1][obj_idx], tokenizer)
             mask = masks[word[0][obj_idx]][idx].squeeze(0)
             ca_map_obj = attn_map[:, :, adj_idx].sum(-1).reshape(b, H, W)
             activation_value = (ca_map_obj * mask).reshape(b, -1).sum(dim=-1)/ca_map_obj.reshape(b, -1).sum(dim=-1)
             obj_loss = obj_loss + torch.mean(3- 4 * activation_value)
        loss += obj_loss

    
    loss = loss / (len(word[1]) * len(attn_maps))
  

    return loss




def update_latent_TR(model, latents1, latents2, text_embeddings1, text_embeddings2, text_embeddings3,controller, t, i,max_iter,scale):
    latents2 = latents2.requires_grad_(True)

    iteration3 = 0

    loss3_ngscale= 5.5
    loss3_scale=scale
    loss3_max_iter =max_iter

 
    mask = torch.zeros((1,1,64,64)).cuda()
    for j in range(len(controller.word[1])):
      if j==0 or (j>=1 and controller.word[0][j]!=controller.word[0][j-1] ):
        mask += controller.adj_mask[controller.word[0][j]][3]

    if i < 25:

        while iteration3 < loss3_max_iter:
        
            # calculate positive loss
            noise_prediction = model.unet(latents1.detach(), t, encoder_hidden_states=text_embeddings1)["sample"]
            noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings2)["sample"]
            atten = controller.step_atten['up_cross'][0:3] + controller.step_atten['down_cross'][4:6]
            loss3 = compute_ca_TR(atten, controller.adj_mask, controller.word, controller.prompt)
            loss = loss3 * loss3_scale

            
            grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents2])[0]
            sigma = sqrt((1 - model.scheduler.alphas_cumprod[t]) / model.scheduler.alphas_cumprod[t])
            latent_unmask = latents2.detach().clone() * (1 - mask)
            latent_mask = latents2 * mask
            latent_mask = latent_mask - grad_cond * sigma * mask
            latents2 = latent_unmask + latent_mask

            iteration3 += 1
            torch.cuda.empty_cache()

    
    return latents2.detach()


@torch.no_grad()
def text2image_ldm_stable_TR(
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
    max_iter=15,
    scale=2.5
):
    batch_size = len(prompt)
    ptp_utils.register_attention_control(model, controller)

   
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
        

   
    model.scheduler.set_timesteps(num_inference_steps)

    latent, latents = ptp_utils.init_latent(latent, model, height, width, generator, batch_size)
    latents1 = latent.detach().clone().to(device)
    latents2 = latent.detach().clone().to(device)
    
    
    max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    
    mask = torch.zeros(1,1,64,64).cuda()
    for i in range(len(controller.word[1])):
      if i==0 or (i>=1 and controller.word[0][i]!=controller.word[0][i-1] ):
        mask += controller.adj_mask[controller.word[0][i]][3]
   
    mask =  max_pool(mask)
    
    for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):

        if i < 25:

            controller.optimizing=True
           
            with torch.enable_grad():
              latent_optimize_new =update_latent_TR(model, latents1, latents2,  text_embeddings1, text_embeddings2,None,controller, t,i,max_iter=max_iter,scale=scale)

            latents2=latent_optimize_new
        
        controller.optimizing=False
        
        if uncond_embeddings_ is None:
            context1 = torch.cat([uncond_embeddings[i].expand(*text_embeddings1.shape), text_embeddings1])
            context2 = torch.cat([uncond_embeddings[i].expand(*text_embeddings2.shape), text_embeddings2])

        else:

            context1 = torch.cat([uncond_embeddings_[0:1], text_embeddings1])
            context2 = torch.cat([uncond_embeddings_[1:2], text_embeddings2])

        
        latents1,latents2 = ptp_utils.diffusion_step2(model, controller,latents1, context1,latents2,context2, t, guidance_scale, low_resource=False)

        if 50-i <= 15:
            latents2 = latents2 * mask + latents1 * (1-mask)
        torch.cuda.empty_cache()

    if return_type == 'image':
        image = ptp_utils.latent2image(model.vae, torch.cat([latents1,latents2]))
    else:
        image = latents
    return image, latent


    

        
    
    

