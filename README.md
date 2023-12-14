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








<h3> Other Applications</h3>  
<p align="center">
<img src="assets/other_apps.jpg"/>  
<br>

<h3> Qualitative Comparison </h3>
<p align="center">
  <table align="center"   style="text-align:center;">
    <tr style="background-color: #F5F5F5">
      <td align="center">
       Simplified <br>Prompt
      </td>
      <td align="center">
       Source Image
      </td>
      <td  align="center">
        <b>MAG-Edit(Ours)</b>
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
      <tr>
      <td style="padding:0;" align="center">
        Denim <br>pants
      </td>
      <td style="width: 105px; height:105px;padding:0;" align="center">
        <img src="assets/compare/p2ppnp/2/source.png" style="width: 100px; height: 100px;margin:0;padding=0;vertical-align:middle;" hspace="0" vspace="0">
      </td>
      <td style="width:105px; height: 105px;padding:0;" align="center">
        <img src="assets/compare/p2ppnp/2/ours.png" style="width: 100px; height: 100px;margin:0;padding=0;vertical-align:middle;" hspace="0" vspace="0">
      </td>
      <td style="width: 105px; height: 105px;padding:0;" align="center">
        <img src="assets/compare/p2ppnp/2/p2p.png" style="width: 100px; height: 100px;margin:0;padding=0;vertical-align:middle;" hspace="0" vspace="0">
      </td>          
      <td style="width: 105px; height: 105px;padding:0;" align="center">
        <img src="assets/compare/p2ppnp/2/pnp.png" style="width: 100px; height: 100px;margin:0;padding=0;vertical-align:middle;" hspace="0" vspace="0">
      </td>
    </tr>
      <tr>
      <td style="padding:0;" align="center">
        Vintgae <br>car
      </td>
      <td style="width: 105px; height:105px;padding:0;" align="center">
        <img src="assets/compare/p2ppnp/3/source.png" style="width: 100px; height: 100px;margin:0;padding=0;vertical-align:middle;" hspace="0" vspace="0">
      </td>
      <td style="width:105px; height: 105px;padding:0;" align="center">
        <img src="assets/compare/p2ppnp/3/ours.png" style="width: 100px; height: 100px;margin:0;padding=0;vertical-align:middle;" hspace="0" vspace="0">
      </td>
      <td style="width: 105px; height: 105px;padding:0;" align="center">
        <img src="assets/compare/p2ppnp/3/p2p.png" style="width: 100px; height: 100px;margin:0;padding=0;vertical-align:middle;" hspace="0" vspace="0">
      </td>          
      <td style="width: 105px; height: 105px;padding:0;" align="center">
        <img src="assets/compare/p2ppnp/3/pnp.png" style="width: 100px; height: 100px;margin:0;padding=0;vertical-align:middle;" hspace="0" vspace="0">
      </td>
    </tr>
      <tr>
      <td style="padding:0;" align="center">
        Slices of <br>steak
      </td>
      <td style="width: 105px; height:105px;padding:0;" align="center">
        <img src="assets/compare/p2ppnp/4/source.png" style="width: 100px; height: 100px;margin:0;padding=0;vertical-align:middle;" hspace="0" vspace="0">
      </td>
      <td style="width:105px; height: 105px;padding:0;" align="center">
        <img src="assets/compare/p2ppnp/4/ours.png" style="width: 100px; height: 100px;margin:0;padding=0;vertical-align:middle;" hspace="0" vspace="0">
      </td>
      <td style="width: 105px; height: 105px;padding:0;" align="center">
        <img src="assets/compare/p2ppnp/4/p2p.png" style="width: 100px; height: 100px;margin:0;padding=0;vertical-align:middle;" hspace="0" vspace="0">
      </td>          
      <td style="width: 105px; height: 105px;padding:0;" align="center">
        <img src="assets/compare/p2ppnp/4/pnp.png" style="width: 100px; height: 100px;margin:0;padding=0;vertical-align:middle;" hspace="0" vspace="0">
      </td>
    </tr>
    <tr style="background-color: #F5F5F5">
      <td align="center">
       Simplified <br>Prompt
      </td>
      <td align="center">
       Source Image
      </td>
      <td  align="center">
        <b>MAG-Edit(Ours)</b>
      </td>
      <td align="center">
        Blended Latent Diffusion
      </td>
      <td  align="center">
        DiffEdit
      </td>
    </tr>
    <tr>
      <td style="padding:0;" align="center">
        Yellow <br>chair
      </td>
      <td style="width: 105px; height:105px;padding:0;" align="center">
        <img src="assets/compare/blend/1/source.png" style="width: 100px; height: 100px;margin:0;padding=0;vertical-align:middle;" hspace="0" vspace="0">
      </td>
      <td style="width:105px; height: 105px;padding:0;" align="center">
        <img src="assets/compare/blend/1/ours.png" style="width: 100px; height: 100px;margin:0;padding=0;vertical-align:middle;" hspace="0" vspace="0">
      </td>
      <td style="width: 105px; height: 105px;padding:0;" align="center">
        <img src="assets/compare/blend/1/blended.png" style="width: 100px; height: 100px;margin:0;padding=0;vertical-align:middle;" hspace="0" vspace="0">
      </td>          
      <td style="width: 105px; height: 105px;padding:0;" align="center">
        <img src="assets/compare/blend/1/iedit.png" style="width: 100px; height: 100px;margin:0;padding=0;vertical-align:middle;" hspace="0" vspace="0">
      </td>
    </tr>
      <tr>
      <td style="padding:0;" align="center">
        Plaid <br>shirt
      </td>
      <td style="width: 105px; height:105px;padding:0;" align="center">
        <img src="assets/compare/blend/2/source.png" style="width: 100px; height: 100px;margin:0;padding=0;vertical-align:middle;" hspace="0" vspace="0">
      </td>
      <td style="width:105px; height: 105px;padding:0;" align="center">
        <img src="assets/compare/blend/2/ours.png" style="width: 100px; height: 100px;margin:0;padding=0;vertical-align:middle;" hspace="0" vspace="0">
      </td>
      <td style="width: 105px; height: 105px;padding:0;" align="center">
        <img src="assets/compare/blend/2/blended.png" style="width: 100px; height: 100px;margin:0;padding=0;vertical-align:middle;" hspace="0" vspace="0">
      </td>          
      <td style="width: 105px; height: 105px;padding:0;" align="center">
        <img src="assets/compare/blend/2/iedit.png" style="width: 100px; height: 100px;margin:0;padding=0;vertical-align:middle;" hspace="0" vspace="0">
      </td>
    </tr>
      <tr>
      <td style="padding:0;" align="center">
        White <br>bird
      </td>
      <td style="width: 105px; height:105px;padding:0;" align="center">
        <img src="assets/compare/blend/3/source.png" style="width: 100px; height: 100px;margin:0;padding=0;vertical-align:middle;" hspace="0" vspace="0">
      </td>
      <td style="width:105px; height: 105px;padding:0;" align="center">
        <img src="assets/compare/blend/3/ours.png" style="width: 100px; height: 100px;margin:0;padding=0;vertical-align:middle;" hspace="0" vspace="0">
      </td>
      <td style="width: 105px; height: 105px;padding:0;" align="center">
        <img src="assets/compare/blend/3/blended.png" style="width: 100px; height: 100px;margin:0;padding=0;vertical-align:middle;" hspace="0" vspace="0">
      </td>          
      <td style="width: 105px; height: 105px;padding:0;" align="center">
        <img src="assets/compare/blend/3/iedit.png" style="width: 100px; height: 100px;margin:0;padding=0;vertical-align:middle;" hspace="0" vspace="0">
      </td>
    </tr>
      <tr>
         <td style="padding:0;" align="center">
        Strawberry
      </td>
      <td style="width: 105px; height:105px;padding:0;" align="center">
        <img src="assets/compare/blend/4/source.png" style="width: 100px; height: 100px;margin:0;padding=0;vertical-align:middle;" hspace="0" vspace="0">
      </td>
      <td style="width:105px; height: 105px;padding:0;" align="center">
        <img src="assets/compare/blend/4/ours.png" style="width: 100px; height: 100px;margin:0;padding=0;vertical-align:middle;" hspace="0" vspace="0">
      </td>
      <td style="width: 105px; height: 105px;padding:0;" align="center">
        <img src="assets/compare/blend/4/blended.png" style="width: 100px; height: 100px;margin:0;padding=0;vertical-align:middle;" hspace="0" vspace="0">
      </td>          
      <td style="width: 105px; height: 105px;padding:0;" align="center">
        <img src="assets/compare/blend/4/iedit.png" style="width: 100px; height: 100px;margin:0;padding=0;vertical-align:middle;" hspace="0" vspace="0">
      </td>
    </tr> 
    <tr>
      <td align="center" style="background-color: #F5F5F5">
       Simplified <br>Prompt
      </td>
      <td align="center">
       Source Image
      </td>
      <td  align="center">
        <b>MAG-Edit(Ours)</b>
      </td>
      <td align="center">
        InstructPix2Pix
      </td>
      <td  align="center">
        MagicBrush
      </td>
    </tr>
    <tr>
      <td style="padding:0;" align="center">
        Yellow <br>chair
      </td>
      <td style="width: 105px; height:105px;padding:0;" align="center">
        <img src="assets/compare/training/1/source.png" style="width: 100px; height: 100px;margin:0;padding=0;vertical-align:middle;" hspace="0" vspace="0">
      </td>
      <td style="width:105px; height: 105px;padding:0;" align="center">
        <img src="assets/compare/training/1/ours.png" style="width: 100px; height: 100px;margin:0;padding=0;vertical-align:middle;" hspace="0" vspace="0">
      </td>
      <td style="width: 105px; height: 105px;padding:0;" align="center">
        <img src="assets/compare/training/1/instruct.png" style="width: 100px; height: 100px;margin:0;padding=0;vertical-align:middle;" hspace="0" vspace="0">
      </td>          
      <td style="width: 105px; height: 105px;padding:0;" align="center">
        <img src="assets/compare/training/1/magic.png" style="width: 100px; height: 100px;margin:0;padding=0;vertical-align:middle;" hspace="0" vspace="0">
      </td>
    </tr>   
      <tr>
      <td style="padding:0;" align="center">
        Yellow <br>chair
      </td>
      <td style="width: 105px; height:105px;padding:0;" align="center">
        <img src="assets/compare/training/2/source.png" style="width: 100px; height: 100px;margin:0;padding=0;vertical-align:middle;" hspace="0" vspace="0">
      </td>
      <td style="width:105px; height: 105px;padding:0;" align="center">
        <img src="assets/compare/training/2/ours.png" style="width: 100px; height: 100px;margin:0;padding=0;vertical-align:middle;" hspace="0" vspace="0">
      </td>
      <td style="width: 105px; height: 105px;padding:0;" align="center">
        <img src="assets/compare/training/2/instruct.png" style="width: 100px; height: 100px;margin:0;padding=0;vertical-align:middle;" hspace="0" vspace="0">
      </td>          
      <td style="width: 105px; height: 105px;padding:0;" align="center">
        <img src="assets/compare/training/2/magic.png" style="width: 100px; height: 100px;margin:0;padding=0;vertical-align:middle;" hspace="0" vspace="0">
      </td>
    </tr>  
    <tr>
      <td style="padding:0;" align="center">
        Yellow <br>chair
      </td>
      <td style="width: 105px; height:105px;padding:0;" align="center">
        <img src="assets/compare/training/3/source.png" style="width: 100px; height: 100px;margin:0;padding=0;vertical-align:middle;" hspace="0" vspace="0">
      </td>
      <td style="width:105px; height: 105px;padding:0;" align="center">
        <img src="assets/compare/training/3/ours.png" style="width: 100px; height: 100px;margin:0;padding=0;vertical-align:middle;" hspace="0" vspace="0">
      </td>
      <td style="width: 105px; height: 105px;padding:0;" align="center">
        <img src="assets/compare/training/3/instruct.png" style="width: 100px; height: 100px;margin:0;padding=0;vertical-align:middle;" hspace="0" vspace="0">
      </td>          
      <td style="width: 105px; height: 105px;padding:0;" align="center">
        <img src="assets/compare/training/3/magic.png" style="width: 100px; height: 100px;margin:0;padding=0;vertical-align:middle;" hspace="0" vspace="0">
      </td>
    </tr>    
  </table>





<!--
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
-->



<h2> Citation </h2>



