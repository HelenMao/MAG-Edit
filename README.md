<h1> MAG-Edit </h1>

This repository is the official implementation of MAG-Edit.

MAG-Edit: Localized Image Editing in Complex Scenarios via
Mask-Based Attention-Adjusted Guidance  
<br/>
[Qi Mao](https://sites.google.com/view/qi-mao/),  
[Lan Chen](), 
[Yuchao Gu](https://ycgu.site/), 
[Zhen Fang](),
[Mike Zheng Shou](https://sites.google.com/view/showlab)
<br/>


[![Project Website](https://img.shields.io/badge/Project-Website-orange
)](https://orannue.github.io/MAG-Edit/)
[![arXiv](https://img.shields.io/badge/arXiv-XXXXX-red
)]()

<p align="center">
<img src="assets/teaser.png"width="1080px"/>  
<br>
<em> (a) <a href="https://github.com/omriav/blended-latent-diffusion">Blended latent diffusion</a>  (b) <a href="https://arxiv.org/abs/2210.11427">DiffEdit</a>  (c) <a href="https://github.com/google/prompt-to-prompt">Prompt2Prompt</a> <br> 
(d)  <a href="https://github.com/MichalGeyer/plug-and-play">Plug-and-play</a>  (e) P2P+Blend (f) PnP+Blend</em>
</p>


## TODO:

- [ ] Release Code
- [ ] Release MAG-Edit paper and project page


<h2> Results </h2>

<p align="center">
<h3> Various Editing Types </h3>
<p align="center">
<img src="assets/editing_types.png"/>  
</p>


<p align="center">
  <table align="center"   style="text-align:center;">
    <tr>
      <td align="center">
       Simplified <br>Prompt
      </td>
      <td align="center">
       Source Image
      </td>
      <td  align="center">
        MAG-Edit(Ours)
      </td>
      <td align="center">
        P2P
      </td>
      <td  align="center">
        PnP
      </td>
    </tr>
    <tr>
      <td style="padding:0;" align="center">
        Green <br>pillow
      </td>
      <td style="width: 105px; height:105px;padding:0;" align="center">
        <img src="assets/compare/p2ppnp/1/source.png" style="width: 100px; height: 100px;margin:0;padding=0;vertical-align:middle;" hspace="0" vspace="0">
      </td>
      <td style="width:105px; height: 105px;padding:0;" align="center">
        <img src="assets/compare/p2ppnp/1/ours.png" style="width: 100px; height: 100px;margin:0;padding=0;vertical-align:middle;" hspace="0" vspace="0">
      </td>
      <td style="width: 105px; height: 105px;padding:0;" align="center">
        <img src="assets/compare/p2ppnp/1/p2p.png" style="width: 100px; height: 100px;margin:0;padding=0;vertical-align:middle;" hspace="0" vspace="0">
      </td>          
      <td style="width: 105px; height: 105px;padding:0;" align="center">
        <img src="assets/compare/p2ppnp/1/pnp.png" style="width: 100px; height: 100px;margin:0;padding=0;vertical-align:middle;" hspace="0" vspace="0">
      </td>
    </tr>
  </table>





<h3> Other Applications</h3>  
<p align="center">
<img src="assets/other_apps.jpg"/>  
<br>

<h3> Qualitative Comparison </h3>
<font size=4>Comparison with <a href="https://github.com/omriav/blended-latent-diffusion">Blended LD</a> and <a href="https://arxiv.org/abs/2210.11427">DiffEdit</a></font>
</p>
<p align="center">
<img src="assets/qualitative_cmp/mask.png"/>  
</p>

<p align="center">
<font size=4>Comparison with <a href="https://github.com/google/prompt-to-prompt">P2P</a> and <a href="https://github.com/MichalGeyer/plug-and-play">PnP</a></font>
</p>
<p align="center">
<img src="assets/qualitative_cmp/p2ppnp.png"/>  
</p>

<p align="center">
<font size=4>Comparison with <a href="https://github.com/timothybrooks/instruct-pix2pix">InstructPix2Pix</a> and <a href="https://github.com/OSU-NLP-Group/MagicBrush">MagicBrush</a></font>
</p>
<p align="center">
<img src="assets/qualitative_cmp/instructimagic.png"/>  
</p>

<h3> Various Editing Scenarios </h3>
<p align="center">
<img src="assets/editing_scenarios.png"/>  
</p>




<h2> Citation </h2>



