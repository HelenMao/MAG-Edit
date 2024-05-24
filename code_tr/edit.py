import os

from network import *
from argparse import ArgumentParser


def run_and_displayloss_TR(prompts, controller, latent=None, run_baseline=False, generator=None, uncond_embeddings=None,
                           verbose=True, i=0, name=None,max_iter=15,scale=2.5):
    images, x_t = text2image_ldm_stable_TR(ldm_stable, prompts, controller, latent=latent,
                                           num_inference_steps=NUM_DDIM_STEPS, guidance_scale=GUIDANCE_SCALE,
                                           generator=generator, uncond_embeddings=uncond_embeddings,max_iter=max_iter,scale=scale)
   
    utils.save_image(images[1], "{}/{}.png".format(name, prompts[1]), nrow=1)
    return images, x_t


null_inversion = NullInversion(ldm_stable)

parser = ArgumentParser()

parser.add_argument("--source_prompt", default="there is a room with champagne sofa chairs table lamp and carpet", type=str)
parser.add_argument("--target_prompt", default="there is a room with green sofa chairs table lamp and carpet", type=str)
parser.add_argument("--target_word", default="green", nargs="+", type=str)
parser.add_argument("--negative_prompt", default=None, type=str)
parser.add_argument("--negative_word", default=None, nargs="+", type=str)
parser.add_argument("--img_path", default="example/1/1.jpg", type=str) 
parser.add_argument("--mask_path", default="example/1/mask.png", type=str)
parser.add_argument("--result_dir", default="result", type=str)
parser.add_argument("--max_iteration", default=15, type=int)
parser.add_argument("--scale", default=2.5, type=float)
args = parser.parse_args()

if __name__ == '__main__':
    (image_gt, image_enc), x_t, uncond_embeddings = null_inversion.invert(args.img_path, args.source_prompt, offsets=(0, 0, 0, 0), verbose=True)
   
    prompts = [args.source_prompt, args.target_prompt]
   
    cross_replace_steps = {'default_': .8, }
    self_replace_steps = 0.5

    positive_word = [[args.mask_path for p in args.target_word], [p for p in args.target_word]]
    negative_word=None
    print("Prompt for Inversion:", args.source_prompt)
    print("Prompt for Editing:", args.target_prompt)
    print("Mask Path:",  positive_word[0])
    print("Positive words:", positive_word[1])



    controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps, None, name=args.result_dir,
                                 word=positive_word, adj_mask=args.mask_path, word_ng=negative_word)
    os.makedirs(args.result_dir,exist_ok=True)
    
    images, _ = run_and_displayloss_TR(prompts, controller, run_baseline=False, latent=x_t,
                                           uncond_embeddings=uncond_embeddings, name=args.result_dir,max_iter=args.max_iteration,scale=args.scale)