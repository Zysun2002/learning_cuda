# include <torch/extension.h>

//检查cuda类型和contiguous
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// 函数声明
torch::Tensor trilinear_fw_cu(
    torch::Tensor feats,
    torch::Tensor points
);


torch::Tensor trilinear_bw_cu(
    const torch::Tensor dL_feat_interp,
    const torch::Tensor feats,
    const torch::Tensor points
);