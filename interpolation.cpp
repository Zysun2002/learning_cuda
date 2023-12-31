# include <torch/extension.h>
# include "utils.h"
// python和cuda之间的桥梁

torch::Tensor trilinear_interpolation_fw(
    torch::Tensor feats,
    torch::Tensor points
){    
    // 检查输入
    CHECK_INPUT(feats);
    CHECK_INPUT(points);

    return trilinear_fw_cu(feats, points);
}

torch::Tensor trilinear_interpolation_bw(
    const torch::Tensor dL_feat_interp,
    const torch::Tensor feats,
    const torch::Tensor points
){    
    CHECK_INPUT(dL_feat_interp);
    CHECK_INPUT(feats);
    CHECK_INPUT(points);

    return trilinear_bw_cu(dL_feat_interp, feats, points);
}

// 允许调用的函数名和函数实体
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("trilinear_interpolation_fw", &trilinear_interpolation_fw);
    m.def("trilinear_interpolation_bw", &trilinear_interpolation_bw);
}

