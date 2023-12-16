### MAG-Edit: Localized Image Editing in Complex Scenarios via Mask-Based Attention-Adjusted Guidance  

This repository is the official implementation of MAG-Edit.

[Qi Mao](https://sites.google.com/view/qi-mao/), [Lan Chen](), [Yuchao Gu](https://ycgu.site/), [Zhen Fang](), [Mike Zheng Shou](https://sites.google.com/view/showlab)


[![Project Website](https://img.shields.io/badge/Project-Website-orange
)](https://orannue.github.io/MAG-Edit/)
[![arXiv](https://img.shields.io/badge/arXiv-XXXXX-red
)]()

<p align="center">
<img src="assets/teaser.png"width="1060px"/>  
<br>
<em> (a) <a href="https://github.com/omriav/blended-latent-diffusion">Blended latent diffusion</a>  (b) <a href="https://arxiv.org/abs/2210.11427">DiffEdit</a>  (c) <a href="https://github.com/google/prompt-to-prompt">Prompt2Prompt</a> <br> 
(d)  <a href="https://github.com/MichalGeyer/plug-and-play">Plug-and-play</a>  (e) P2P+Blend (f) PnP+Blend</em>
</p>

## :bookmark: Abstract
<b>TL; DR: <font color="red">MAG-Edit</font> is the first method specifically designed to
address localized image editing in complex scenarios without training.</b>

<details><summary>CLICK for the full abstract</summary>
Recent diffusion-based image editing approaches have exhibited impressive editing capabilities in images with simple compositions. However, localized editing in complex scenarios has not been well-studied in the literature, despite its growing real-world demands. Existing mask-based inpainting methods fall short of retaining the underlying structure within the edit region. Meanwhile, mask-free attention-based methods often exhibit editing leakage and misalignment in more complex compositions. In this work, we develop MAG-Edit, a training-free, inference-stage optimization method, which enables localized image editing in complex scenarios. In particular, MAG-Edit optimizes the noise latent feature in diffusion models by maximizing two mask-based cross-attention constraints of the edit token, which in turn gradually enhances the local alignment with the desired prompt. Extensive quantitative and qualitative experiments demonstrate the effectiveness of our method in achieving both text alignment and structure preservation for localized editing within complex scenarios.
</details>

## :pencil: Changelog
- 2023.12.19 Release Project Page and Paper!
## 💡TODO:

- [ ] Release Code
- [x] Release MAG-Edit paper and project page


<p align="center">
<h2> Various Editing Types </h2>
<p align="center">
<img src="assets/editing_types.png"/>  
</p>

<h2> Other Applications</h2>  
<p align="center">
<img src="assets/other_apps.jpg"/>  
<br>

<h2> Qualitative Comparison </h2>
<font size=4>Comparison with training-free methods</font>

<p align="center">
  <table align="center"   style="text-align:center;">
    <tr >
      <td align="center" style="width: 10%;" >
       Simplified <br>Prompt
      </td>
      <td align="center" style="width: 15%;">
       Source <br> Image
      </td>
      <td  align="center" style="width: 15%;">
        <b>Ours</b>
      </td>
      <td align="center" style="width: 15%;">
       <a href="https://github.com/omriav/blended-latent-diffusion">Blended <br> LD</a>
      </td>
      <td  align="center" style="width: 15%;">
      <a href="https://arxiv.org/abs/2210.11427">DiffEdit</a>
      </td>
      <td  align="center" style="width: 15%;">
      <a herf="https://github.com/google/prompt-to-prompt">P2P</a>
      </td>
      <td  align="center" style="width: 15%;">
      <a herf="https://github.com/MichalGeyer/plug-and-play">PnP</a>
      </td>
    </tr>
    <tr>
      <td  align="center" style="width: 10%;" >
        Green <br>pillow
      </td>
      <td  align="center" style="width: 15%;">
        <img src="assets/compare/training-free/1/source.png" width=70%>
      </td>
      <td align="center" style="width: 15%;"> 
        <img src="assets/compare/training-free/1/ours.png" style="width:15%">
      </td>
      <td  align="center" style="width: 15%;">
        <img src="assets/compare/training-free/1/blended.png" width=70%>
      </td>          
      <td  align="center" style="width: 15%;">
        <img src="assets/compare/training-free/1/diffedit.png" width=70%>
      </td>
      <td  align="center" style="width: 15%;">
        <img src="assets/compare/training-free/1/p2p.png" width=70%>
      </td>      
      <td  align="center" style="width: 15%;">
        <img src="assets/compare/training-free/1/pnp.png" width=70%>
      </td>     
    </tr>
    <tr>
      <td  align="center" style="width: 10%;">
        Denim <br>pants
      </td>
      <td align="center" style="width: 15%;">
        <img src="assets/compare/training-free/2/source.png" width=70%>
      </td>
      <td  align="center" style="width: 15%;">
        <img src="assets/compare/training-free/2/ours.png" width=70%>
      </td>
      <td  align="center" style="width: 15%;">
        <img src="assets/compare/training-free/2/blended.png" width=70%>
      </td>          
      <td  align="center" style="width: 15%;">
        <img src="assets/compare/training-free/2/diffedit.png" width=70%>
      </td>
      <td  align="center" style="width: 15%;">
        <img src="assets/compare/training-free/2/p2p.png" width=70%>
      </td>      
      <td align="center">
        <img src="assets/compare/training-free/2/pnp.png" width=70%>
      </td>     
    </tr>
    <tr>
      <td  align="center" style="width: 10%;">
        White <br>bird
      </td>
      <td  align="center" style="width: 15%;">
        <img src="assets/compare/training-free/3/source.png" width=70%>
      </td>
      <td  align="center" style="width: 15%;">
        <img src="assets/compare/training-free/3/ours.png" width=70%>
      </td>
      <td  align="center" style="width: 15%;">
        <img src="assets/compare/training-free/3/blended.png" width=70%>
      </td>          
      <td  align="center" style="width: 15%;">
        <img src="assets/compare/training-free/3/diffedit.png" width=70%>
      </td>
      <td  align="center" style="width: 15%;">
        <img src="assets/compare/training-free/3/p2p.png" width=70%>
      </td>      
      <td align="center" style="width: 15%;">
        <img src="assets/compare/training-free/3/pnp.png" width=70%>
      </td>     
    </tr>
    <tr>
      <td align="center" style="width: 10%;">
        Slices of <br>steak
      </td>
      <td  align="center" style="width: 15%;">
        <img src="assets/compare/training-free/4/source.png" width=70%>
      </td>
      <td align="center" style="width: 15%;">
        <img src="assets/compare/training-free/4/ours.png"  width=70%>
      </td>
      <td align="center" style="width: 15%;">
        <img src="assets/compare/training-free/4/blended.png"  width=70%>
      </td>          
      <td  align="center" style="width: 15%;">
        <img src="assets/compare/training-free/4/diffedit.png" width=70%>
      </td>
      <td align="center" style="width: 15%;">
        <img src="assets/compare/training-free/4/p2p.png"  width=70%>
      </td>      
      <td align="center" style="width: 15%;">
        <img src="assets/compare/training-free/4/pnp.png" width=70%>
      </td>     
  </table>







>
<!--
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



## :triangular_flag_on_post: Citation 

```
@article{qi2023MAG-Edit,
      title={MAG-Edit: Localized Image Editing in Complex Scenarios via Mask-Based Attention-Adjusted Guidance  }, 
      author={Qi Mao and Lan Chen and Yuchao Gu and Zhen Fang and Mike Zheng Shou},
      year={2023},
      journal={arXiv:XXXXX},
}
``` 


## :revolving_hearts: Acknowledgements

This repository borrows heavily from [prompt-to-prompt](https://github.com/google/prompt-to-prompt/). Thanks to the authors for sharing their code and models.




