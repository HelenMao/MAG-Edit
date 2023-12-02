# MAG-Edit

This repository is the official implementation of MAG-Edit.

MAG-Edit: Localized Image Editing in Complex Scenarios via
Mask-Based Attention-Adjusted Guidance  

<p align="center">
<img src="assets/teaser.png" width="1080px"/>  
<br>
<em>Given a source image, a source prompt, an edited prompt and a mask as input, our method, MAG-Edit, generates an image that aligns with the edited prompt in the masked region.</em>
</p>

 

## TODO:
- [ ] Release Code
- [ ] Release MAG-Bench



## Results

### Various Editing Scenarios

<table class="center">
 <tr>
  <td style="text-align:center;" colspan="4"><b>Indoor Scenario</b></td>
</tr>
<tr>
  <td style="text-align:center;"><b>Input Image</b></td>
  <td style="text-align:center;" colspan="3"><b>Output Image</b></td>
</tr>
<tr>
  <td><img src="assets/editing_scenarios/indoor/source.jpg"></td>
  <td><img src="assets/editing_scenarios/indoor/sofa.png"></td>
  <td><img src="assets/editing_scenarios/indoor/table.png"></td>              
  <td><img src="assets/editing_scenarios/indoor/carpet.png"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;color:gray;"></td>
  <td width=25% style="text-align:center;">"Blue and velvet sofa”</td>
  <td width=25% style="text-align:center;">"Marble table"</td>
  <td width=25% style="text-align:center;">"Yellow and damask carpet"</td>
</tr>
 <tr>
  <td style="text-align:center;" colspan="4"><b>Outdoor Scenario</b></td>
</tr>
<tr>
  <td style="text-align:center;"><b>Input Image</b></td>
  <td style="text-align:center;" colspan="3"><b>Output Image</b></td>
</tr>
<tr>
  <td><img src="assets/editing_scenarios/outdoor/1/source.jpg"></td>
  <td><img src="assets/editing_scenarios/outdoor/1/hat1.png"></td>
  <td><img src="assets/editing_scenarios/outdoor/1/hat2.png"></td>              
  <td><img src="assets/editing_scenarios/outdoor/1/grass.png"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;color:gray;"></td>
  <td width=25% style="text-align:center;">"Pirate hat”</td>
  <td width=25% style="text-align:center;">"Tall chef hat"</td>
  <td width=25% style="text-align:center;">"Leaves-covered grass"</td>
</tr>
<tr>
  <td><img src="assets/editing_scenarios/outdoor/2/source.jpg"></td>
  <td><img src="assets/editing_scenarios/outdoor/2/limousine.png"></td>
  <td><img src="assets/editing_scenarios/outdoor/2/roadster.png"></td>              
  <td><img src="assets/editing_scenarios/outdoor/2/graffiti.png"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;color:gray;"></td>
  <td width=25% style="text-align:center;">"Limousine”</td>
  <td width=25% style="text-align:center;">"Roadster"</td>
  <td width=25% style="text-align:center;">"Graffiti"</td>
</tr>



### Various Editing Types

<p align="center">
<img src="assets/editing_types.png" width="1080px"/>  



### Controllable Granularity  Localized Editing  









## MAG-Bench









## Citation