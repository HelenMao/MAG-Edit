# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm.notebook import tqdm
from torchvision import utils
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from math import sqrt
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
from diffusers.schedulers.scheduling_ddim  import DDIMSchedulerOutput


class AttentionBase:
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def after_step(self):
        pass

    def __call__(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads,scale,**kwargs):
        self.scale=scale
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
def view_images(images, num_rows=1, offset_ratio=0.02,name=None):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0
    
    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    pil_img.save(name)
 
    
def diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False):
    
    if low_resource:
        noise_pred_uncond = model.unet(latents.detach(), t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents]*2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = controller.step_callback(latents,t)
    

    return latents



def diffusion_step2(model, controller, latents1, context1,latents2,context2, t, guidance_scale, low_resource=False):

    


    


    latents_input1 = torch.cat([latents1]*2)
    noise_pred1 = model.unet(latents_input1, t, encoder_hidden_states=context1)["sample"]
    noise_pred_uncond, noise_prediction_text = noise_pred1.chunk(2)
    noise_pred1 = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)

   
    latents_input2 = torch.cat([latents2]*2)
    noise_pred2 = model.unet(latents_input2, t, encoder_hidden_states=context2)["sample"]
    noise_pred_uncond, noise_prediction_text = noise_pred2.chunk(2)
    noise_pred2 = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)

    #noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents1 = model.scheduler.step(noise_pred1, t, latents1)["prev_sample"]
    latents1 = controller.step_callback(latents1,t)
    
    latents2 = model.scheduler.step(noise_pred2, t, latents2)["prev_sample"]
    latents2 = controller.step_callback(latents2,t)

    return latents1,latents2




def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    #utils.save_image(image, "{}.png".format(i), nrow=1)
    #image = image.cpu().permute(0, 2, 3, 1).numpy()
    #image = (image * 255).astype(np.uint8)
    return image


def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(batch_size,  model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents




def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, context=None, **kwargs):
            batch_size, sequence_length, dim = x.shape
            
            h = self.heads
            q = self.to_q(x)
            
            # <<< FIX CONTEXT ASSIGNMENT >>>
            # Original check:
            # is_cross = context is not None
            # context = context if is_cross else x
            
            # Revised check:
            # Check if text embeddings were passed via kwargs instead of context argument
            if context is None and 'encoder_hidden_states' in kwargs:
                is_cross = True
                # Retrieve encoder_hidden_states from kwargs
                context = kwargs['encoder_hidden_states'] 
            else:
                 # Standard handling (context argument might be None for self-attn)
                is_cross = context is not None
                context = context if is_cross else x
                
            # <<< ADD DEBUG PRINT >>>
            # print(f"Place: {place_in_unet}, Is Cross: {is_cross}, X Shape: {x.shape}, Context Shape: {context.shape if context is not None else 'None'}, Kwargs Keys: {list(kwargs.keys())}")

            k = self.to_k(context)
            v = self.to_v(context)
            q = self.reshape_heads_to_batch_dim(q)
            k = self.reshape_heads_to_batch_dim(k)
            v = self.reshape_heads_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
          
            
            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            attn = controller(attn, is_cross, place_in_unet)
    
           
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.reshape_batch_dim_to_heads(out)
            return to_out(out)

        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if 'Attention' in net_.__class__.__name__:
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count
  #print("word_place",word_place)
    #print("split_text",split_text)
    
def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
   
    elif type(word_place) is int:
        word_place = [word_place]
    #print("word_place",word_place)
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0
        #print("words_encode",words_encode)
        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def update_alpha_time_word(alpha, bounds: Union[float, Tuple[float, float]], prompt_ind: int,
                           word_inds: Optional[torch.Tensor]=None):
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[: start, prompt_ind, word_inds] = 0
    alpha[start: end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(prompts, num_steps,
                                   cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]],
                                   tokenizer, max_num_words=77):
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0., 1.)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"],
                                                  i)
    for key, item in cross_replace_steps.items():
        if key != "default_":
             inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
             for i, ind in enumerate(inds):
                 if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words)
    return alpha_time_words
