# include <torch/extension.h>
# include "utils.h"

template <typename scalar_t>
__global__ void trilinear_bw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dL_dfeat_interp,
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> feats,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> points,
    torch:: PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> dL_dfeats
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int f = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (n > feats.size(0) || f > feats.size(2)) return;

    const scalar_t u = (points[n][0]+1)/2;
    const scalar_t v = (points[n][1]+1)/2;
    const scalar_t w = (points[n][2]+1)/2;

    const scalar_t a = (1-v)*(1-w);
    const scalar_t b = (1-v)*w;
    const scalar_t c = v*(1-w);
    const scalar_t d = 1-a-b-c;

    dL_dfeats[n][0][f] = (1-u)*a*dL_dfeat_interp[n][f];
    dL_dfeats[n][1][f] = (1-u)*b*dL_dfeat_interp[n][f];
    dL_dfeats[n][2][f] = (1-u)*c*dL_dfeat_interp[n][f];
    dL_dfeats[n][3][f] = (1-u)*d*dL_dfeat_interp[n][f];
    dL_dfeats[n][4][f] = u*a*dL_dfeat_interp[n][f];
    dL_dfeats[n][5][f] = u*b*dL_dfeat_interp[n][f];
    dL_dfeats[n][6][f] = u*c*dL_dfeat_interp[n][f];
    dL_dfeats[n][7][f] = u*d*dL_dfeat_interp[n][f];
}

//每个kernal的动作内容
template <typename scalar_t>
__global__ void trilinear_fw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> feats,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> points,
    torch:: PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> feat_interp
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int f = blockIdx.y * blockDim.y + threadIdx.y;
    // 取得每一个thread编号（n方向和f方向）
    
    //如果当前kernal不工作：
    if (n > feats.size(0) || f > feats.size(2)) return;

    //计算距离
    const scalar_t u = (points[n][0]+1)/2;
    const scalar_t v = (points[n][1]+1)/2;
    const scalar_t w = (points[n][2]+1)/2;

    //计算权重
    const scalar_t a = (1-v)*(1-w);
    const scalar_t b = (1-v)*w;
    const scalar_t c = v*(1-w);
    const scalar_t d = 1-a-b-c;

    feat_interp[n][f] = (1-u)*(a*feats[n][0][f] +
                            b*feats[n][1][f] +
                            c*feats[n][2][f] +
                            d*feats[n][3][f]) + 
                        u*(a*feats[n][4][f] +
                        b*feats[n][5][f] +
                        c*feats[n][6][f] +
                        d*feats[n][7][f]);
}

torch::Tensor trilinear_fw_cu(
    torch::Tensor feats,
    torch::Tensor points
){
    //先生成空的结果值，然后填充
    const int N = feats.size(0), F = feats.size(2);
    
    // 为每个kernal配置参数
    torch::Tensor feat_interp = torch::zeros({N, F}, feats.options());
    // feat_interp的加载设备、数据类型和feats一致

    const dim3 threads(16, 16); //每个thread包含一个(16, 16)的区域
    const dim3 blocks ((N+threads.x)/threads.x, (F+threads.y-1)/threads.y);
    // 密铺所有的thread

    //启动各个kernal
    AT_DISPATCH_FLOATING_TYPES(feats.type(), "trilinear_fw_cu",
    ([&] {
        trilinear_fw_kernel<scalar_t><<<blocks, threads>>>(
            feats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            points.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            feat_interp.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return feat_interp;
}

torch::Tensor trilinear_bw_cu(
    const torch::Tensor dL_d_feat_interp,
    const torch::Tensor feats,
    const torch::Tensor points
){
    const int N = feats.size(0), F = feats.size(2);
    
    torch::Tensor dl_dfeats = torch::zeros({N, 8, F}, feats.options());

    const dim3 threads(16, 16); 
    const dim3 blocks ((N+threads.x)/threads.x, (F+threads.y-1)/threads.y);

    //启动各个kernal
    AT_DISPATCH_FLOATING_TYPES(feats.type(), "trilinear_fw_cu",
    ([&] {
        trilinear_bw_kernel<scalar_t><<<blocks, threads>>>(
            dL_d_feat_interp.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            feats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            points.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            dl_dfeats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return dl_dfeats;

}