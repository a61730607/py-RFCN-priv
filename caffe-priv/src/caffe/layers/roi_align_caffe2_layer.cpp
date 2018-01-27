#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/roi_align_caffe2_layer.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void ROIAlignCaffe2Layer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ROIAlignCaffe2Parameter roi_align_caffe2_param = this->layer_param_.roi_align_caffe2_param();
  CHECK_GT(roi_align_caffe2_param.pooled_h(), 0)
      << "pooled_h must be > 0";
  CHECK_GT(roi_align_caffe2_param.pooled_w(), 0)
      << "pooled_w must be > 0";
  pooled_height_ = roi_align_caffe2_param.pooled_h();
  pooled_width_ = roi_align_caffe2_param.pooled_w();
  spatial_scale_ = roi_align_caffe2_param.spatial_scale();
  sampling_ratio_ = roi_align_caffe2_param.sampling_ratio();
  LOG(INFO) << "Spatial scale: " << spatial_scale_;
}

template <typename Dtype>
void ROIAlignCaffe2Layer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->Reshape(bottom[1]->num(), channels_, pooled_height_,
      pooled_width_);
}

template <typename Dtype>
void ROIAlignCaffe2Layer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	NOT_IMPLEMENTED;
  
}

template <typename Dtype>
void ROIAlignCaffe2Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	NOT_IMPLEMENTED;
  
}


#ifdef CPU_ONLY
STUB_GPU(ROIAlignCaffe2Layer);
#endif

INSTANTIATE_CLASS(ROIAlignCaffe2Layer);
REGISTER_LAYER_CLASS(ROIAlignCaffe2);

}  // namespace caffe
