import argparse
import torch
import numpy as np
from numpy import *

# compute rollout between attention layers
def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                          for i in range(len(all_layer_matrices))]
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer+1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)
    return joint_attention

class LRP:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_LRP(self, input, index=None, method="transformer_attribution", is_ablation=False, start_layer=0):
        output = self.model(input)
        kwargs = {"alpha": 1}
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        return self.model.relprop(torch.tensor(one_hot_vector).to(input.device), method=method, is_ablation=is_ablation,
                                  start_layer=start_layer, **kwargs)



class Baselines:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_cam_attn(self, input, index=None, mae=False):
        output = self.model(input.cuda(), register_hook=True)
        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        #################### attn
        grad = self.model.blocks[-1].attn.get_attn_gradients()
        cam = self.model.blocks[-1].attn.get_attention_map()
        if mae:
            cam = cam[0, :, 1:, 1:]
            grad = grad[0, :, 1:, 1:]
            grad = grad.mean(dim=[0, 2], keepdim=True)
            cam = (cam * grad).mean(0).mean(1).clamp(min=0)
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = cam[0, :, 0, 1:].reshape(-1, 14, 14)
            grad = grad[0, :, 0, 1:].reshape(-1, 14, 14)
            grad = grad.mean(dim=[1, 2], keepdim=True)
            cam = (cam * grad).mean(0).clamp(min=0)
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam
        #################### attn
    
    def generate_attn(self, input, mae=False, dino=False):
        output = self.model(input.cuda(), register_hook=True)
        cam = self.model.blocks[-1].attn.get_attention_map()
        if mae:
            cam = cam[0, :, 1:, 1:]
            cam = cam.mean(0).mean(1).clamp(min=0)
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = cam[0, :, 0, 1:].reshape(-1, 14, 14)
            cam = cam.mean(0).clamp(min=0)
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam

    def generate_rollout(self, input, start_layer=0, mae=False):
        self.model(input)
        blocks = self.model.blocks
        all_layer_attentions = []
        for blk in blocks:
            attn_heads = blk.attn.get_attention_map()
            avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
            all_layer_attentions.append(avg_heads)
        rollout = compute_rollout_attention(all_layer_attentions, start_layer=start_layer)
        if mae:
            return rollout[:,1:,1:].mean(1)
        else:
            return rollout[:,0, 1:]
    
    def generate_ours(self, input, index=None, steps=20, start_layer=4, samples=20, noise=0.2, mae=False, dino=False, ssl=False):
        b = input.shape[0]
        output = self.model(input, register_hook=True)
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((b, output.size()[-1]), dtype=np.float32)
        one_hot[np.arange(b), index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)
        
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        
        _, num_head, num_tokens, _ = self.model.blocks[-1].attn.get_attention_map().shape

        R = torch.eye(num_tokens, num_tokens).expand(b, num_tokens, num_tokens).cuda()
        for nb, blk in enumerate(self.model.blocks):
            if nb < start_layer-1:
                continue
            
            grad = blk.attn.get_attn_gradients()
            grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
            cam = blk.attn.get_attention_map()
            cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
            
            Ih = torch.mean(torch.matmul(cam.transpose(-1,-2), grad).abs(), dim=(-1,-2))
            Ih = Ih/torch.sum(Ih)
            cam = torch.matmul(Ih,cam.reshape(num_head,-1)).reshape(num_tokens,num_tokens)
            
            R = R + torch.matmul(cam.cuda(), R.cuda())
        
        if ssl:
            if mae:
                return R[:, 1:, 1:].abs().mean(axis=1)
            elif dino:
                return (R[:, 1:, 1:].abs().mean(axis=1)+R[:, 0, 1:].abs())
            else:
                return R[:, 0, 1:].abs()
            
#         avg head
#         R = torch.eye(num_tokens, num_tokens).expand(b, num_tokens, num_tokens).cuda()
#         for nb, blk in enumerate(self.model.blocks):
#             if nb < start_layer-1:
#                 continue
#             cam = blk.attn.get_attention_map()
#             cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
#             cam = cam.mean(0)
            
#             R += torch.matmul(cam.cuda(), R.cuda())
            
            #last one
#         R = torch.eye(num_tokens, num_tokens).expand(b, num_tokens, num_tokens).cuda()
#         blk = self.model.blocks[-1]
            
#         grad = blk.attn.get_attn_gradients()
#         grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
#         cam = blk.attn.get_attention_map()
#         cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])

#         Ih = torch.mean(torch.matmul(cam.transpose(-1,-2), grad).abs(), dim=(-1,-2))
#         Ih = Ih/torch.sum(Ih)
#         cam = torch.matmul(Ih,cam.reshape(num_head,-1)).reshape(num_tokens,num_tokens)
            
#         R += torch.matmul(cam.cuda(), R.cuda())
        
            
#         last gradient 
#         gradients = self.model.blocks[-1].attn.get_attn_gradients()
#         W_state = (gradients / steps).clamp(min=0).mean(1).reshape(b, num_tokens, num_tokens)
#         R = W_state * R
        
#         Smooth gradient
#         total_gradients = torch.zeros(b, num_head, num_tokens, num_tokens).cuda()
#         for i in range(samples):        
#             # forward propagation
#             dev = noise*(torch.max(input)-torch.min(input))
#             noise = torch.normal(0.0, 0.3859, [1, 3, 224, 224], dtype=torch.float32).cuda()
#             data_perturbed = input+noise

#             # backward propagation
#             output = self.model(data_perturbed, register_hook=True)
#             one_hot = np.zeros((b, output.size()[-1]), dtype=np.float32)
#             one_hot[np.arange(b), index] = 1
#             one_hot_vector = one_hot
#             one_hot = torch.from_numpy(one_hot).requires_grad_(True)
#             one_hot = torch.sum(one_hot.cuda() * output)

#             self.model.zero_grad()
#             one_hot.backward(retain_graph=True)

#             # cal grad
#             gradients = self.model.blocks[-1].attn.get_attn_gradients()
#             total_gradients += gradients        
       
#         W_state = (total_gradients / steps).clamp(min=0).mean(1)[:, 0, :].reshape(b, 1, num_tokens)
            
#         R = W_state * R    
        
        total_gradients = torch.zeros(b, num_head, num_tokens, num_tokens).cuda()
        for alpha in np.linspace(0, 1, steps):        
            # forward propagation
            data_scaled = input * alpha

            # backward propagation
            output = self.model(data_scaled, register_hook=True)
            one_hot = np.zeros((b, output.size()[-1]), dtype=np.float32)
            one_hot[np.arange(b), index] = 1
            one_hot_vector = one_hot
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            one_hot = torch.sum(one_hot.cuda() * output)

            self.model.zero_grad()
            one_hot.backward(retain_graph=True)

            # cal grad
            gradients = self.model.blocks[-1].attn.get_attn_gradients()
            total_gradients += gradients        
       
        W_state = (total_gradients / steps).clamp(min=0).mean(1).reshape(b, num_tokens, num_tokens)
        R = W_state * R
        
        if mae:
            return R[:, 1:, 1:].mean(axis=1)
        elif dino:
            return (R[:, 1:, 1:].mean(axis=1) + R[:, 0, 1:])
        else:
            return R[:, 0, 1:]
    
    
    def generate_ig(self, input, index=None, steps=20, start_layer=6, samples=20, noise=0.2):
        b = input.shape[0]
        output = self.model(input, register_hook=True)
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((b, output.size()[-1]), dtype=np.float32)
        one_hot[np.arange(b), index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)
        
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        
        _, num_head, num_tokens, _ = self.model.blocks[-1].attn.get_attention_map().shape
        
        total_gradients = torch.zeros([b, 1, 224, 224]).cuda()
        for alpha in np.linspace(0, 1, steps):        
            # forward propagation
            data_scaled = input * alpha

            # backward propagation
            output = self.model(data_scaled, register_hook=True)
            one_hot = np.zeros((b, output.size()[-1]), dtype=np.float32)
            one_hot[np.arange(b), index] = 1
            one_hot_vector = one_hot
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            one_hot = torch.sum(one_hot.cuda() * output)

            self.model.zero_grad()
            one_hot.backward(retain_graph=True)

            # cal grad
            gradients = self.model.get_input_gradients().sum(1).unsqueeze(1)
            total_gradients += gradients        
       
        W_state = total_gradients / steps
        
        return W_state
    
    def generate_sg(self, input, index=None, steps=20, start_layer=6, samples=20, noise=0.2):
        b = input.shape[0]
        output = self.model(input, register_hook=True)
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((b, output.size()[-1]), dtype=np.float32)
        one_hot[np.arange(b), index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)
        
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        
        _, num_head, num_tokens, _ = self.model.blocks[-1].attn.get_attention_map().shape
        
        total_gradients = torch.zeros([b, 1, 224, 224]).cuda()
        
        for alpha in np.linspace(0, 1, steps): 
            dev = noise*(torch.max(input)-torch.min(input))
            noise = torch.normal(0.0, 0.3859, [1, 3, 224, 224], dtype=torch.float32).cuda()
            data_perturbed = input+noise

            # backward propagation
            output = self.model(data_perturbed, register_hook=True)
            one_hot = np.zeros((b, output.size()[-1]), dtype=np.float32)
            one_hot[np.arange(b), index] = 1
            one_hot_vector = one_hot
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            one_hot = torch.sum(one_hot.cuda() * output)

            self.model.zero_grad()
            one_hot.backward(retain_graph=True)

            # cal grad
            gradients = self.model.get_input_gradients().sum(1).unsqueeze(1)
            total_gradients += gradients        
       
        W_state = total_gradients / steps
        
        return W_state
    
    def generate_ours_c(self, input, index=None, steps=20, start_layer=6, samples=20, noise=0.2, mae=False, ssl=False, dino=False):
        b = input.shape[0]
        output = self.model(input, register_hook=True)
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((b, output.size()[-1]), dtype=np.float32)
        one_hot[np.arange(b), index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)
        
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        
        _, num_head, num_tokens, _ = self.model.blocks[-1].attn.get_attention_map().shape

        R = torch.eye(num_tokens, num_tokens).expand(b, num_tokens, num_tokens).cuda()
        for nb, blk in enumerate(self.model.blocks):
            if nb < start_layer-1:
                continue
            z = blk.get_input()
            vproj = blk.attn.get_vproj()
    
            order = torch.linalg.norm(vproj, dim=-1).squeeze()/torch.linalg.norm(z, dim=-1).squeeze()
            m = torch.diag_embed(order)
            cam = blk.attn.get_attention_map()
            cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1]).mean(0)
            
            R = R + torch.matmul(torch.matmul(cam.cuda(), m.cuda()), R.cuda())
        
        if ssl:
            if mae:
                return R[:, 1:, 1:].abs().mean(axis=1)
            elif dino:
                return (R[:, 1:, 1:].abs().mean(axis=1)+R[:, 0, 1:].abs())
            else:
                return R[:, 0, 1:].abs()
        
        total_gradients = torch.zeros(b, num_head, num_tokens, num_tokens).cuda()
        for alpha in np.linspace(0, 1, steps):        
            # forward propagation
            data_scaled = input * alpha

            # backward propagation
            output = self.model(data_scaled, register_hook=True)
            one_hot = np.zeros((b, output.size()[-1]), dtype=np.float32)
            one_hot[np.arange(b), index] = 1
            one_hot_vector = one_hot
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            one_hot = torch.sum(one_hot.cuda() * output)

            self.model.zero_grad()
            one_hot.backward(retain_graph=True)

            # cal grad
            gradients = self.model.blocks[-1].attn.get_attn_gradients()
            total_gradients += gradients        
       
        W_state = (total_gradients / steps).clamp(min=0).mean(1).reshape(b, num_tokens, num_tokens)
            
        R = W_state * R.abs()     
        
        if mae:
            return R[:, 1:, 1:].mean(axis=1)
        elif dino:
            return (R[:, 1:, 1:].mean(axis=1) + R[:, 0, 1:])
        else:
            return R[:, 0, 1:]
    
    
    def generate_genattr(self, input, start_layer=1, index=None, mae=False):
        b = input.shape[0]
        output = self.model(input, register_hook=True)
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((b, output.size()[-1]), dtype=np.float32)
        one_hot[np.arange(b), index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)
        
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        
        _, num_head, num_tokens, _ = self.model.blocks[-1].attn.get_attention_map().shape

        R = torch.eye(num_tokens, num_tokens).expand(b, num_tokens, num_tokens).cuda()
        for nb, blk in enumerate(self.model.blocks):
            if nb < start_layer-1:
                continue
                
            cam = blk.attn.get_attention_map()
            cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
            grad = blk.attn.get_attn_gradients()
            grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
            cam = (grad*cam).mean(0).clamp(min=0)
            R = R + torch.matmul(cam, R)  
        
        if mae:
            return R[:, 1:, 1:].mean(axis=1)
        else:
            return R[:, 0, 1:]
