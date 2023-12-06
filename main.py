import torch
import mycuda   #_ 调用mycuda

# - python中的类将forward和backward封装起来
class Trilinear_interpolation_cuda(torch.autograd.Function):
    @ staticmethod
    def forward(ctx, feats, points):
        feat_interp = mycuda.trilinear_interpolation_fw(feats, points)
        
        # - ctx中保存backward可能需要用到的参数
        ctx.save_for_backward(feats, points)
        return feat_interp
    def backward(ctx, dL_feat_interp):
        feats, points = ctx.saved_tensors
        dL_dfeats = mycuda.trilinear_interpolation_bw(dL_feat_interp.contiguous(), feats, points)

        #_ 这里的dL_dfeats和None分别对应forward中的feats和points的梯度
        return dL_dfeats, None

if __name__ == '__main__':
    N = 1024; F = 256;
    feats = torch.rand(N, 8, F, device='cuda').requires_grad_()
    points = torch.rand(N, 3, device='cuda')*2-1

    out_cuda = Trilinear_interpolation_cuda.apply(feats, points)
    torch.cuda.synchronize()
    loss = out_cuda.sum()
    loss.backward()

    print(feats.grad.shape)
