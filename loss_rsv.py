def update_latent4(model, latents1, latents2,  text_embeddings1, text_embeddings2, text_embeddings3, controller, t,i):
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
 
    
    loss3_ngscale = 2.5
    loss1_ngscale = 5.0
    loss2_ngscale = 2.0
    
    loss1_max_iter=15
    loss2_max_iter=15
    loss3_max_iter=15

    
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

    
    if i<25:
       
        loss3 = compute_ca_loss3(atten, controller.adj_mask,controller.word, controller.prompt,controller.name)
        print("firstloss3",loss3.item())
        while loss3.item()  > loss3_threshold and iteration3 < loss3_max_iter:
       
          # positive
          atten = controller.step_atten['up_cross'][0:3]+controller.step_atten['down_cross'][4:6]
          loss3 = compute_ca_loss3(atten, controller.adj_mask,controller.word, controller.prompt,controller.name)
          loss = loss3*loss3_scale
          print("loss3:",loss3.item())
          
          '''if controller.word_ng is not None:
            noise_prediction = model.unet(latents1, t, encoder_hidden_states=text_embeddings1)["sample"]
            noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings3)["sample"]   
            atten = controller.step_atten['up_cross'][0:3]+controller.step_atten['down_cross'][4:6]
            loss3 = compute_ca_loss3(atten, controller.adj_mask,controller.word_ng,controller.ng_prompt,controller.name)
            loss = loss -1.0 * loss3_ngscale * loss3        
            print("loss3_ng",loss3.item())'''
                          
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
         
          atten = controller.step_atten['up_cross'][0:3]+controller.step_atten['down_cross'][4:6]
          loss3 = compute_ca_loss3(atten, controller.adj_mask,controller.word, controller.prompt,controller.name)       
     
        
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
            noise_prediction = model.unet(latents1.detach(), t, encoder_hidden_states=text_embeddings1)["sample"]
            noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings2)["sample"]              
            iteration2 += 1
           

            # positive
            atten = aggregate_attention( attention_store=controller,\
                                                  res=16,\
                                                  from_where=("up", "down"),\
                                                  is_cross=True,\
                                                  select=0)

            loss2 = compute_ca_loss2(atten, controller.adj_mask,controller.word, controller.prompt,controller.name)
            loss = loss2                    
            print("Loss2:",loss2.item())  
            
            '''if controller.word_ng is not None:
                noise_prediction = model.unet(latents1, t, encoder_hidden_states=text_embeddings1)["sample"]
                noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings3)["sample"]               
                atten = aggregate_attention( attention_store=controller,\
                                                    res=16,\
                                                    from_where=("up", "down"),\
                                                    is_cross=True,\
                                                    select=0)'''
        
                loss2 = compute_ca_loss2(atten, controller.adj_mask, controller.word_ng,controller.ng_prompt,controller.name)
                print("loss2_ng",loss2.item())
              
                loss =  loss -1.0 *  loss2_ngscale * loss2                              
             
            grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents2], retain_graph=True)[0]
            latent_unmask = latents2.detach().clone()* (1-mask)
            latent_mask = latents2 * mask
            latent_mask = latent_mask - step_size * grad_cond * mask
            latents2 = latent_unmask + latent_mask
                

            #with torch.no_grad():
            noise_prediction = model.unet(latents1.detach(), t, encoder_hidden_states=text_embeddings1)["sample"]
            noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings2)["sample"]   
            atten = aggregate_attention( attention_store=controller,\
                                            res=16,\
                                            from_where=("up", "down"),\
                                            is_cross=True,\
                                            select=0)
            loss2 = compute_ca_loss2(atten, controller.adj_mask, controller.word,controller.prompt,controller.name)
            if iteration2 >= loss2_max_iter:
                break
                
    if i < 25:   
       atten = controller.step_atten['up_cross'][0:3]+controller.step_atten['down_cross'][4:6]   
       loss1 = compute_ca_loss1(atten, controller.adj_mask, controller.word,controller.prompt,controller.name)
       while loss1.item()  > loss1_threshold and iteration1 < loss1_max_iter:
               
                #positive
                atten = controller.step_atten['up_cross'][0:3]+controller.step_atten['down_cross'][4:6]
                loss1 = compute_ca_loss1(atten, controller.adj_mask, controller.word,controller.prompt, controller.name)
                loss = loss1*loss1_scale
                print("loss1:",loss1.item())
                
                #negative
                '''if controller.word_ng is not None:
                    noise_prediction = model.unet(latents1, t, encoder_hidden_states=text_embeddings1)["sample"]
                    noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings3)["sample"]                    
                    atten = controller.step_atten['up_cross'][0:3]+controller.step_atten['down_cross'][4:6]
                    loss1_ngscale = compute_ca_loss1(atten, controller.adj_mask,controller.word_ng,controller.ng_prompt,controller.name)                
                    loss =  loss-1.0* loss1_ngscale *  loss1
                    print("loss1_ng",loss1.item())'''
                grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents2])[0]
                sigma = sqrt((1-model.scheduler.alphas_cumprod[t])/model.scheduler.alphas_cumprod[t])
                latent_unmask = latents2.detach().clone()* (1-mask)
                latent_mask = latents2 * mask
                latent_mask = latent_mask - grad_cond  * mask
                latents2 = latent_unmask + latent_mask
                iteration1 += 1
                
                noise_prediction = model.unet(latents1.detach(), t, encoder_hidden_states=text_embeddings1)["sample"]
                noise_prediction = model.unet(latents2, t, encoder_hidden_states=text_embeddings2)["sample"]   
                atten = controller.step_atten['up_cross'][0:3]+controller.step_atten['down_cross'][4:6]
                loss1 = compute_ca_loss1(atten, controller.adj_mask, controller.word,controller.prompt,controller.name)        
                torch.cuda.empty_cache() 
    return latents2.detach()      