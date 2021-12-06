import torch as th
import torch.nn.functional as F
import sys

class MILNCELoss(th.nn.Module):
    def __init__(self):
        super(MILNCELoss, self).__init__()
    
    def forward(self, video_embd, text_embd):
        num_vids = len(video_embd)
        video_embd = video_embd.view(-1, 1, video_embd.size(-1))
        text_embd = th.reshape(text_embd, (-1, text_embd.size(-1), 1))
        scores = th.bmm(video_embd, text_embd)
        scores = scores.squeeze()
        scores = scores.view(num_vids, -1)
        nominator = scores * th.eye(scores.shape[0])[:, :,].cuda()
        nominator = nominator.sum(-1)
        nominator = nominator.unsqueeze(-1)
        nominator = th.logsumexp(nominator, dim=-1)
        
        denominator = th.cat((scores, scores.permute(1, 0)), dim=-1)
        denominator = th.logsumexp(denominator, dim=-1)
        loss = th.mean(denominator - nominator)
        
        return loss