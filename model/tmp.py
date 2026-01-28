for k in range():

mask_mat = torch.eq(labels, labels.T).float().to(device)
logits_mask = torch.scatter(torch.ones_like(mask_mat), 1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)

mask_mat = mask_mat * logits_mask

dot_prod_mat = torch.div(torch.matmul(features, features.T), self.temperature)


logits_max, _ = torch.max(dot_prod_mat, dim=1, keepdim=True)
dot_prod_mat = dot_prod_mat - logits_max.detach()

exp_mat = torch.exp(dot_prod_mat) * logits_mask
exp_mat_sum = exp.mat.sum(1)

exp_pos_mat = torch.exp(dot_prod_mat) * mask_mat
exp_pos_mat_sum = exp_pos_mat.sum(1)

log_prob = dot_prod_mat - torch.log(exp_mat.sum(1, keepdim=True))