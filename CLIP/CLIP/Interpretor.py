import numpy as np
import torch
import copy

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


class Interpretor:
    def __init__(self, model, device):
        self.model = model
        self.model.eval()
        self.device = device

    
    def generate_ours(self, image, texts, steps=20, num_layers=10, index=None):
        
        batch_size = image.shape[0]
        logits_per_image, logits_per_text = self.model(image, texts, backward=True)
        probs = logits_per_image.softmax(dim=-1).detach()
        if index is None:
            values, indices = probs[0].topk(1)
            index=indices.item()

        one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
        one_hot[:, index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * logits_per_image)
        self.model.zero_grad()

        image_attn_blocks = list(dict(self.model.visual.transformer.resblocks.named_children()).values())
        num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
        h = len(image_attn_blocks)
        R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(self.device)
        R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
        for i, blk in enumerate(image_attn_blocks):
            if i <=num_layers:
              continue
            grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
            cam = blk.attn_probs.detach()
            cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
            grad = grad.reshape(batch_size ,-1, grad.shape[-1], grad.shape[-1])

            Ih = torch.mean(torch.matmul(cam.transpose(-1,-2), grad).abs(), dim=(-1,-2)).unsqueeze(0)
            Ih = Ih/torch.sum(Ih)
            cam = torch.matmul(Ih.reshape(batch_size,1,h) ,cam.reshape(batch_size,h,-1)).reshape(batch_size,num_tokens,num_tokens)

            cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
            R = R + torch.bmm(cam, R)

        total_gradients = torch.zeros(batch_size, h, num_tokens, num_tokens).cuda()
        for alpha in np.linspace(0, 1, steps):        
            # forward propagation
            images_scaled = image * alpha

            # backward propagation
            logits_per_image, logits_per_text = self.model(images_scaled, texts, backward=True)
            one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
            one_hot[np.arange(batch_size), index] = 1
            one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            one_hot = torch.sum(one_hot.cuda() * logits_per_image)
            self.model.zero_grad()

            grad = torch.autograd.grad(one_hot, [image_attn_blocks[-1].attn_probs], retain_graph=True)[0].detach()
            grad = grad.reshape(batch_size ,-1, grad.shape[-1], grad.shape[-1])

            total_gradients += grad        

        W_state = (total_gradients / steps).clamp(min=0).mean(1)[:, 0, :].reshape(batch_size, 1, num_tokens)

        R = W_state * R  

        return R[:,0,1:]
    
    def generate_ours_c(self, image, texts, steps=20, num_layers=8):
        
        batch_size = texts.shape[0]
        images = image.repeat(batch_size, 1, 1, 1)
        logits_per_image, logits_per_text = self.model(images, texts, backward=True)
        probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
        index = [i for i in range(batch_size)]
        one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
        one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * logits_per_image)
        self.model.zero_grad()

        image_attn_blocks = list(dict(self.model.visual.transformer.resblocks.named_children()).values())
        num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
        h = len(image_attn_blocks)
        R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(self.device)
        R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
        for i, blk in enumerate(image_attn_blocks):
            if i <=num_layers:
              continue
            cam = blk.attn_probs.detach()
            cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1]).mean(1)
            
            z = blk.get_input().transpose(0,1)
            vproj = blk.get_vproj().transpose(0,1)
            
            order = torch.linalg.norm(vproj, dim=-1).squeeze()/torch.linalg.norm(z, dim=-1).squeeze()
            m = torch.diag_embed(order).cuda()

            R = R + torch.bmm(torch.matmul(cam, m), R)

        total_gradients = torch.zeros(batch_size, h, num_tokens, num_tokens).cuda()
        for alpha in np.linspace(0, 1, steps):        
            # forward propagation
            images_scaled = images * alpha

            # backward propagation
            logits_per_image, logits_per_text = self.model(images_scaled, texts, backward=True)
            one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
            one_hot[np.arange(batch_size), index] = 1
            one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            one_hot = torch.sum(one_hot.cuda() * logits_per_image)
            self.model.zero_grad()

            grad = torch.autograd.grad(one_hot, [image_attn_blocks[-1].attn_probs], retain_graph=True, allow_unused=True)[0].detach()
            grad = grad.reshape(batch_size ,-1, grad.shape[-1], grad.shape[-1])

            total_gradients += grad        

        W_state = (total_gradients / steps).clamp(min=0).mean(1)[:, 0, :].reshape(batch_size, 1, num_tokens)

        R = W_state * R  

        return R[:,0,1:]
    
    
    def generate_cam_attn(self, image, texts):
        
        batch_size = texts.shape[0]
        images = image.repeat(batch_size, 1, 1, 1)
        logits_per_image, logits_per_text = self.model(images, texts, backward=True)
        probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
        index = [i for i in range(batch_size)]
        one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
        one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * logits_per_image)
        self.model.zero_grad()
        
        image_attn_blocks = list(dict(self.model.visual.transformer.resblocks.named_children()).values())
        blk = image_attn_blocks[-1]
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(batch_size ,-1, grad.shape[-1], grad.shape[-1])
        
        cam = cam[:, :, 0, 1:].reshape(batch_size, -1, 7,  7)
        grad = grad[:, :, 0, 1:].reshape(batch_size, -1,  7,  7)
        grad = grad.mean(dim=[-1, -2], keepdim=True)
        cam = (cam * grad).mean(1).clamp(min=0)
        cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam
        

    def generate_rollout(self, image, texts, start_layer=0):
        
        batch_size = texts.shape[0]
        images = image.repeat(batch_size, 1, 1, 1)
        logits_per_image, logits_per_text = self.model(images, texts, backward=True)
        
        blocks = list(dict(self.model.visual.transformer.resblocks.named_children()).values())
        all_layer_attentions = []
        for blk in blocks:
            cam = blk.attn_probs.detach()
            cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
            avg_heads = (cam.sum(dim=1) / cam.shape[1]).detach()
            all_layer_attentions.append(avg_heads)
            
        rollout = compute_rollout_attention(all_layer_attentions, start_layer=start_layer)
        return rollout[:,0, 1:]
    
    
    def generate_raw_attention(self, image, texts):
        batch_size = texts.shape[0] 
        images = image.repeat(batch_size, 1, 1, 1)
        logits_per_image, logits_per_text = self.model(images, texts, backward=True)
        blocks = list(dict(self.model.visual.transformer.resblocks.named_children()).values())
        blk = blocks[-1]
        cam = blk.attn_probs.detach()
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        avg_heads = (cam.sum(dim=1) / cam.shape[1]).detach()
           
        return avg_heads[:,0, 1:]
    
    
    def generate_generic_attribution(self, image, texts, num_layers=10):
        
        batch_size = texts.shape[0]
        images = image.repeat(batch_size, 1, 1, 1)
        logits_per_image, logits_per_text = self.model(images, texts, backward=True)
        probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
        index = [i for i in range(batch_size)]
        one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
        one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * logits_per_image)
        self.model.zero_grad()

        image_attn_blocks = list(dict(self.model.visual.transformer.resblocks.named_children()).values())
        num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
        R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(self.device)
        R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
        for i, blk in enumerate(image_attn_blocks):
            if i <=num_layers:
              continue
            grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
            cam = blk.attn_probs.detach()
            cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=1)
            R = R + torch.bmm(cam, R)
        
        image_relevance = R[:, 0, 1:]
        return image_relevance
      