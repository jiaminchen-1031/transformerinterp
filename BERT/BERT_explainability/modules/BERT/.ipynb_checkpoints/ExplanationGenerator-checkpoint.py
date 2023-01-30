import argparse
import numpy as np
import torch
import glob

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

class Generator:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    def generate_LRP(self, input_ids, attention_mask,
                     index=None, start_layer=11):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
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

        self.model.relprop(torch.tensor(one_hot_vector).to(input_ids.device), **kwargs)

        cams = []
        blocks = self.model.bert.encoder.layer
        for blk in blocks:
            grad = blk.attention.self.get_attn_gradients()
            cam = blk.attention.self.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cams.append(cam.unsqueeze(0))
        rollout = compute_rollout_attention(cams, start_layer=start_layer)
        rollout[:, 0, 0] = 0
        return rollout[:, 0]


    def generate_LRP_last_layer(self, input_ids, attention_mask,
                     index=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
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

        self.model.relprop(torch.tensor(one_hot_vector).to(input_ids.device), **kwargs)

        cam = self.model.bert.encoder.layer[-1].attention.self.get_attn_cam()[0]
        cam = cam.clamp(min=0).mean(dim=0).unsqueeze(0)
        cam[:, 0, 0] = 0
        return cam[:, 0]

#     def generate_full_lrp(self, input_ids, attention_mask,
#                      index=None):
#         output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
#         kwargs = {"alpha": 1}

#         if index == None:
#             index = np.argmax(output.cpu().data.numpy(), axis=-1)

#         one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
#         one_hot[0, index] = 1
#         one_hot_vector = one_hot
#         one_hot = torch.from_numpy(one_hot).requires_grad_(True)
#         one_hot = torch.sum(one_hot.cuda() * output)

#         self.model.zero_grad()
#         one_hot.backward(retain_graph=True)

#         cam = self.model.relprop(torch.tensor(one_hot_vector).to(input_ids.device), **kwargs)
#         cam = cam.sum(dim=2)
#         cam[:, 0] = 0
#         return cam

    def generate_attn_last_layer(self, input_ids, attention_mask,
                     index=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        cam = self.model.bert.encoder.layer[-1].attention.self.get_attn()[0]
        cam = cam.mean(dim=0).unsqueeze(0)
        cam[:, 0, 0] = 0
        return cam[:, 0]

    def generate_rollout(self, input_ids, attention_mask, start_layer=0, index=None):
        self.model.zero_grad()
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        blocks = self.model.bert.encoder.layer
        all_layer_attentions = []
        for blk in blocks:
            attn_heads = blk.attention.self.get_attn()
            avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
            all_layer_attentions.append(avg_heads)
        rollout = compute_rollout_attention(all_layer_attentions, start_layer=start_layer)
        rollout[:, 0, 0] = 0
        return rollout[:, 0]

    def generate_attn_gradcam(self, input_ids, attention_mask, index=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
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

        self.model.relprop(torch.tensor(one_hot_vector).to(input_ids.device), **kwargs)

        cam = self.model.bert.encoder.layer[-1].attention.self.get_attn()
        grad = self.model.bert.encoder.layer[-1].attention.self.get_attn_gradients()

        cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
        grad = grad.mean(dim=[1, 2], keepdim=True)
        cam = (cam * grad).mean(0).clamp(min=0).unsqueeze(0)
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        cam[:, 0, 0] = 0
        return cam[:, 0]
    
    def generate_ours(self, input_ids, attention_mask, index=None, steps=20):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
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
        
        blocks = [self.model.bert.encoder.layer[-1]]
        
        _, num_head, num_tokens, _ = self.model.bert.encoder.layer[-1].attention.self.get_attn().shape

        R = torch.eye(num_tokens, num_tokens).expand(1, num_tokens, num_tokens).cuda()
        for blk in blocks:
            grad = blk.attention.self.get_attn_gradients()
            grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
            cam = blk.attention.self.get_attn()
            cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
            
            Ih = torch.mean(torch.matmul(cam.transpose(-1,-2), grad).abs(), dim=(-1,-2))
            Ih = Ih/torch.sum(Ih)
            cam = torch.matmul(Ih,cam.reshape(num_head,-1)).reshape(num_tokens,num_tokens)
            
            R += torch.matmul(cam.cuda(), R.cuda())
        
        total_gradients = torch.zeros(1, num_head, num_tokens, num_tokens).cuda()
        for alpha in np.linspace(0, 1, steps):        

            output = self.model(input_ids=input_ids, attention_mask=attention_mask, alpha=alpha)[0]
            
            one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
            one_hot[0, index] = 1
            one_hot_vector = one_hot
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            one_hot = torch.sum(one_hot.cuda() * output)

            self.model.zero_grad()
            one_hot.backward(retain_graph=True)

            # cal grad
            gradients = blocks[-1].attention.self.get_attn_gradients()
            total_gradients += gradients        
       
        W_state = (total_gradients / steps).clamp(min=0).mean(1)[:, 0, :].reshape(1, 1, num_tokens)
        
#         print('W_state', W_state[:, 0])
#         print('R', R[:, 0])
            
        R = W_state * R
        R[:, 0, 0] = 0

        return R[:, 0]
    
    def generate_ours_c(self, input_ids, attention_mask, index=None, steps=20):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
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
        
        blocks = self.model.bert.encoder.layer[-4:]
        
        _, num_head, num_tokens, _ = self.model.bert.encoder.layer[-1].attention.self.get_attn().shape

        R = torch.eye(num_tokens, num_tokens).expand(1, num_tokens, num_tokens).cuda()
        for blk in blocks:
            cam = blk.attention.self.get_attn()
            cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1]).mean(0)
            
            z = blk.get_input()
            v = blk.attention.self.get_v()
            vproj = torch.matmul(v, blk.attention.output.dense.weight.t())
            
            order = torch.linalg.norm(vproj, dim=-1).squeeze()/torch.linalg.norm(z, dim=-1).squeeze()
            m = torch.diag(order)
            
            R = R + torch.matmul(torch.matmul(cam.cuda(), m.cuda()), R.cuda())
        
        
        ###################################
#         output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
#         kwargs = {"alpha": 1}
        
#         if index == None:
#             index = np.argmax(output.cpu().data.numpy(), axis=-1)

#         one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
#         one_hot[0, index] = 1
#         one_hot_vector = one_hot
#         one_hot = torch.from_numpy(one_hot).requires_grad_(True)
#         one_hot = torch.sum(one_hot.cuda() * output)
        
#         self.model.zero_grad()
#         one_hot.backward(retain_graph=True)
        
#         blocks = self.model.bert.encoder.layer[-2:]
        
#         _, num_head, num_tokens, _ = self.model.bert.encoder.layer[-1].attention.self.get_attn().shape

#         R = torch.eye(num_tokens, num_tokens).expand(1, num_tokens, num_tokens).cuda()
#         for blk in blocks:
#             cam = blk.attention.self.get_attn()
            
# #             cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
            
#             z = blk.get_input()
#             v = blk.attention.self.get_v().transpose(-2, -3)
            
#             proj = torch.cat(torch.split(blk.attention.output.dense.weight.unsqueeze(0), v.shape[-1], dim=-1), dim=0).unsqueeze(0).transpose(-1, -2)
            
#             vproj = torch.matmul(v, proj)
#             order = torch.linalg.norm(vproj, dim=-1).squeeze()/torch.linalg.norm(z, dim=-1).squeeze()
#             m = torch.diag_embed(order).unsqueeze(0)
            
#             R = R + torch.matmul(torch.matmul(cam.cuda(), m.cuda()).sum(1), R.cuda())
            
        #########################################
            
#             vproj = torch.matmul(v, blk.attention.output.dense.weight.t())
            
#             order = torch.linalg.norm(vproj, dim=-1).squeeze()/torch.linalg.norm(z, dim=-1).squeeze()
#             m = torch.diag(order)
            
#             R = R + torch.matmul(torch.matmul(cam.cuda(), m.cuda()), R.cuda())
        
        total_gradients = torch.zeros(1, num_head, num_tokens, num_tokens).cuda()
        for alpha in np.linspace(0, 1, steps):        

            output = self.model(input_ids=input_ids, attention_mask=attention_mask, alpha=alpha)[0]
            
            one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
            one_hot[0, index] = 1
            one_hot_vector = one_hot
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            one_hot = torch.sum(one_hot.cuda() * output)

            self.model.zero_grad()
            one_hot.backward(retain_graph=True)

            # cal grad
            gradients = blocks[-1].attention.self.get_attn_gradients()
            total_gradients += gradients        
       
        W_state = (total_gradients / steps).clamp(min=0).mean(1)[:, 0, :].reshape(1, 1, num_tokens)
        
#         print('W_state', W_state[:, 0])
#         print('R', R[:, 0])
            
        R = W_state * R
        R[:, 0, 0] = 0

        return R[:, 0]
    
    def generate_genattr(self, input_ids, attention_mask, index=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
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

        blocks = [self.model.bert.encoder.layer[-1]]
        num_tokens = blocks[0].attention.self.get_attn().shape[-1]
        R = torch.eye(num_tokens, num_tokens).expand(1, num_tokens, num_tokens).cuda()
        for blk in blocks:
            grad = blk.attention.self.get_attn_gradients()
            grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
            cam = blk.attention.self.get_attn()
            cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
            
            cam = grad * cam
            cam = cam.reshape(1, -1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=1)
            R += torch.matmul(cam.cuda(), R.cuda())
            
        R[:, 0, 0] = 0
        
        return R[:, 0]

