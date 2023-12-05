<h1> MAG-Edit </h1>

This repository is the official implementation of MAG-Edit.

MAG-Edit: Localized Image Editing in Complex Scenarios via
Mask-Based Attention-Adjusted Guidance  

<p align="center">
<img src="assets/teaser.png"width="1080px"/>  
<br>
<em>Given a source image, a source prompt, an edited prompt and a mask as input, our method, MAG-Edit, generates an image that aligns with the edited prompt in the masked region.</em>
</p>


## TODO:

- [ ] Release Code
- [ ] Release Demo


<h2> Results </h2>

<h3> Various Editing Scenarios </h3>
<div>
  <table  align="center" border="1" width="100%" style="max-width: 580px; text-align:center; margin:0 auto">
    <tr>
      <td width="120px" style="vertical-align:middle;text-align:center;" colspan="4"><b>Indoor Scenario</b></td>
    </tr>
    <tr>
      <td  width="120px"style="vertical-align:middle;text-align:center;color:grey;">Input Image</td>
      <td  width="120px"  style="vertical-align:middle;text-align:center;color:grey;" colspan="3">Output Image</td>
    </tr>
    <tr>
      <td  width="120px" align="center" style="vertical-align:middle;"><img src="assets/editing_scenarios/indoor/source.jpg"  style="width: 120px; height: 120px;margin:0;padding=0;"></td>
      <td   width="120px" align="center"><img src="assets/editing_scenarios/indoor/sofa.png" style="width: 120px; height: 120px;margin:0;padding=0;"></td>
      <td  width="120px" align="center"><img src="assets/editing_scenarios/indoor/table.png" style="width: 120px; height: 120px;margin:0;padding=0;"></td>              
      <td  width="120px" align="center"><img src="assets/editing_scenarios/indoor/carpet.png" style="width: 120px; height: 120px;margin:0;padding=0;"></td>
    </tr>
    <tr>
      <td  style="vertical-align:middle;text-align:center;color:grey;">Simplified edited prompt</td>
      <td  style="vertical-align:middle;text-align:center;"><b>Blue and velvet</b> sofa</td>
      <td  style="vertical-align:middle;text-align:center;"><b>Marble</b> table</td>
      <td style="vertical-align:middle;text-align:center;"><b>Yellow and damask</b> carpet</td>
    </tr>
    <tr>
      <td style="vertical-align:middle;text-align:center;" colspan="4"><b>Outdoor Scenario</b></td>
    </tr>
    <tr>
      <td style="text-align:center;color:grey;">Input Image</td>
      <td style="text-align:center;color:grey;" colspan="3">Output Image</td>
    </tr>
    <tr>
      <td align="center"><img src="assets/editing_scenarios/outdoor/1/source.jpg" style="width: 120px; height: 120px;margin:0;padding=0;"></td>
      <td align="center"><img src="assets/editing_scenarios/outdoor/1/hat1.png" style="width: 120px; height: 120px;margin:0;padding=0;"></td>
      <td align="center"><img src="assets/editing_scenarios/outdoor/1/hat2.png" style="width: 120px; height: 120px;margin:0;padding=0;"></td>              
      <td align="center"><img src="assets/editing_scenarios/outdoor/1/grass.png" style="width: 120px; height: 120px;margin:0;padding=0;"></td>
    </tr>
    <tr>
      <td  style="vertical-align:middle;text-align:center;color:gray;">Simplified edited prompt</td>
      <td  style="vertical-align:middle;text-align:center;"><b>Pirate hat</b></td>
      <td  style="vertical-align:middle;text-align:center;"><b>Tall chef hat</b></td>
      <td  style="vertical-align:middle;text-align:center;"><b>Leaves-covered</b> grass</td>
    </tr>
    <tr>
      <td align="center"><img src="assets/editing_scenarios/outdoor/2/source.jpg" style="width: 120px; height: 120px;margin:0;padding=0;"></td>
      <td align="center"><img src="assets/editing_scenarios/outdoor/2/limousine.png" style="width: 120px; height: 120px;margin:0;padding=0;"></td>
      <td align="center"><img src="assets/editing_scenarios/outdoor/2/jeep.png" style="width: 120px; height: 120px;margin:0;padding=0;"></td>              
      <td align="center"><img src="assets/editing_scenarios/outdoor/2/graffiti.png" style="width: 120px; height: 120px;margin:0;padding=0;"></td>
    </tr>
    <tr>
      <td style="vertical-align:middle;text-align:center;color:gray;">Simplified edited prompt</td>
      <td  style="vertical-align:middle;text-align:center;"><b>Limousine</b></td>
      <td style="vertical-align:middle;text-align:center;"><b>Jeep</b></td>
      <td  style="vertical-align:middle;text-align:center;"><b>With graffiti</b></td>
    </tr>
  </table>
  </div>




<h3> Various Editing Types </h3>

  <table align="center" border="1" width="100%" style="max-width: 650px; text-align:center; margin:0 auto">
    <tr>
      <td width="80px" style="vertical-align:middle;text-align:center;color:grey;">Editing Type</td>
      <td  width="80px" style="vertical-align:middle;text-align:center;color:grey;"colspan="2">Object Attribute Manipulation</td>
      <td  width="80px" style="vertical-align:middle;text-align:center;color:grey;">Object Replacement</td>
      <td  width="80px" style="vertical-align:middle;text-align:center;color:grey;">Object Addition</td>
      <td  width="80px" style="vertical-align:middle;text-align:center;color:grey;">Object Removal</td>
    </tr>
    <tr>
      <td width="80px" style="vertical-align:middle;" align="center"><img src="assets/editing_type/1/source.png" style="width: 80px; height: auto;margin:0;padding=0;"></td>
      <td width="80px" style="vertical-align:middle;" align="center"><img src="assets/editing_type/1/colorful.png" style="width: 80px; height: auto;margin:0;padding=0;"></td>
      <td width="80px" style="vertical-align:middle;" align="center"><img src="assets/editing_type/1/redwine.png" style="width: 80px; height: auto;margin:auto;padding=0;"></td>              
      <td width="80px" style="vertical-align:middle;" align="center"><img src="assets/editing_type/1/bowl.png" style="width: 80px; height: 80px;margin:0;padding=0;"></td>
      <td width="80px" style="vertical-align:middle;" align="center"><img src="assets/editing_type/1/butterfly.png" style="width: 80px; height: 80px;margin:0;padding=0;"></td>
      <td width="80px" style="vertical-align:middle;" align="center"><img src="assets/editing_type/1/remove.png" style="width: 80px; height: 80px;margin:0;padding=0;"></td>
    </tr>
    <tr>
      <td  style="vertical-align:middle;text-align:center;color:grey;font-size=5px;">Simplified edited prompt</td>
      <td  style="vertical-align:middle;text-align:center;font-size=5px;"><b>Colorful</b> sheet</td>
      <td  style="vertical-align:middle;text-align:center;font-size=5px;"><b>Red Wine</b></td>
      <td style="vertical-align:middle;text-align:center;font-size=5px;"><b>Bowl</b></td>
      <td style="vertical-align:middle;text-align:center;font-size=5px;"><b>Blue butterfly</b></td>
      <td style="vertical-align:middle;text-align:center;font-size=5px;">Remove a <b>lemon</b></td>      
    </tr>
    <tr>
      <td style="vertical-align:middle;" align="center"><img src="assets/editing_type/2/source.jpg" style="width: 80px; height: 80px;margin:0;padding=0;"></td>
      <td style="vertical-align:middle;" align="center"><img src="assets/editing_type/2/wooden.png" style="width: 80px; height: 80px;margin:0;padding=0;"></td>
      <td style="vertical-align:middle;" align="center"><img src="assets/editing_type/2/lit.png" style="width: 80px; height: 80px;margin:0;padding=0;"></td>              
      <td style="vertical-align:middle;" align="center"><img src="assets/editing_type/2/jack.png" style="width: 80px; height: 80px;margin:0;padding=0;"></td>
      <td style="vertical-align:middle;" align="center"><img src="assets/editing_type/2/curtain.png" style="width: 80px; height: 80px;margin:0;padding=0;"></td>
      <td style="vertical-align:middle;" align="center"><img src="assets/editing_type/2/remove.png" style="width: 80px; height: 80px;margin:0;padding=0;"></td>
    </tr>
    <tr>
      <td  style="vertical-align:middle;text-align:center;color:grey;">Simplified edited prompt</td>
      <td  style="vertical-align:middle;text-align:center;"><b>Wooden house</b></td>
      <td  style="vertical-align:middle;text-align:center;"><b>Lit window</b></td>
      <td style="vertical-align:middle;text-align:center;"><b>Jack-O'-Lantern</b></td>
      <td style="vertical-align:middle;text-align:center;"><b>Curtained</b> window</td>
      <td style="vertical-align:middle;text-align:center;">Remove a <b>flag</b></td>      
    </tr> 
    <tr>
      <td style="vertical-align:middle;" align="center"><img src="assets/editing_type/3/source.jpg" style="width: 80px; height: 80px;margin:0;padding=0;"></td>
      <td style="vertical-align:middle;" align="center"><img src="assets/editing_type/3/creamy.png" style="width: 80px; height: 80px;margin:0;padding=0;"></td>
      <td style="vertical-align:middle;" align="center"><img src="assets/editing_type/3/pinklace.png" style="width: 80px; height: 80px;margin:0;padding=0;"></td>              
      <td style="vertical-align:middle;" align="center"><img src="assets/editing_type/3/bulb.png" style="width: 80px; height: 80px;margin:0;padding=0;"></td>
      <td style="vertical-align:middle;" align="center"><img src="assets/editing_type/3/photos.png" style="width: 80px; height: 80px;margin:0;padding=0;"></td>
      <td style="vertical-align:middle;" align="center"><img src="assets/editing_type/3/remove.png" style="width: 80px; height: 80px;margin:0;padding=0;"></td>
    </tr>
    <tr>
      <td  style="vertical-align:middle;text-align:center;color:grey;">Simplified edited prompt</td>
      <td  style="vertical-align:middle;text-align:center;"><b>Creamy</b> bread</td>
      <td  style="vertical-align:middle;text-align:center;"><b>Pink and lace sheet</b></td>
      <td style="vertical-align:middle;text-align:center;"><b>Light bulb</b></td>
      <td style="vertical-align:middle;text-align:center;"><b>Polaroid photos</b>
</td>
      <td style="vertical-align:middle;text-align:center;">Remove a <b>strawberry</b></td>      
    </tr>  
  </table>





<h3> Controllable Granularity  Localized Editing </h3>  














<h2> Citation </h2>



