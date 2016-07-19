#include <algorithm>
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  moving_average_ = this->layer_param_.bn_param().moving_average();
  bn_momentum_ = this->layer_param_.bn_param().momentum();
  bn_eps_ = this->layer_param_.bn_param().eps();
  // Initialize parameters
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (moving_average_) {
      this->blobs_.resize(4);
    } else {
      this->blobs_.resize(2);
    }
    vector<int> shape;
    shape.push_back(1);
    shape.push_back(bottom[0]->channels());
    // slope
    this->blobs_[0].reset(new Blob<Dtype>(shape));
    shared_ptr<Filler<Dtype> > slope_filler(GetFiller<Dtype>(
        this->layer_param_.bn_param().slope_filler()));
    slope_filler->Fill(this->blobs_[0].get());
    // bias
    this->blobs_[1].reset(new Blob<Dtype>(shape));
    shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
        this->layer_param_.bn_param().bias_filler()));
    bias_filler->Fill(this->blobs_[1].get());
    if (this->blobs_.size() > 2) {
      // running mean
      this->blobs_[2].reset(new Blob<Dtype>(shape));
      caffe_set(this->blobs_[2]->count(), Dtype(0),
          this->blobs_[2]->mutable_cpu_data());
      // running variance
      this->blobs_[3].reset(new Blob<Dtype>(shape));
      caffe_set(this->blobs_[3]->count(), Dtype(0),
          this->blobs_[3]->mutable_cpu_data());
    }
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void BNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  top[0]->ReshapeLike(*(bottom[0]));

  broadcast_buffer_.ReshapeLike(*(bottom[0]));
  spatial_statistic_.Reshape(num_, channels_, 1, 1);
  batch_statistic_.Reshape(1, channels_, 1, 1);

  x_norm_.ReshapeLike(*(bottom[0]));
  x_std_.ReshapeLike(batch_statistic_);

  spatial_sum_multiplier_.Reshape(1, 1, height_, width_);
  caffe_set(spatial_sum_multiplier_.count(), Dtype(1),
      spatial_sum_multiplier_.mutable_cpu_data());
  batch_sum_multiplier_.Reshape(num_, 1, 1, 1);
  caffe_set(batch_sum_multiplier_.count(), Dtype(1),
      batch_sum_multiplier_.mutable_cpu_data());
}

template <typename Dtype>
void BNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  const Dtype* const_bottom_data = bottom[0]->cpu_data();
  const Dtype* const_top_data = top[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  const Dtype* scale_data = this->blobs_[0]->cpu_data();
  const Dtype* shift_data = this->blobs_[1]->cpu_data();

  // Mean normalization
  if (this->phase_ == TEST && moving_average_) {
    // Use the moving average mean
    caffe_copy(batch_statistic_.count(), this->blobs_[2]->cpu_data(),
        batch_statistic_.mutable_cpu_data());
  } else if (this->phase_ == TRAIN) {
    // Compute the mean by averaging over spatial and batch dimensions.
    caffe_cpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_,
        Dtype(1) / (height_ * width_), const_bottom_data,
        spatial_sum_multiplier_.cpu_data(), Dtype(0),
        spatial_statistic_.mutable_cpu_data());
    caffe_cpu_gemv<Dtype>(CblasTrans, num_, channels_,
        Dtype(1) / num_, spatial_statistic_.cpu_data(),
        batch_sum_multiplier_.cpu_data(), Dtype(0),
        batch_statistic_.mutable_cpu_data());
    // Add to the moving average
    if (moving_average_) {
      caffe_cpu_axpby(batch_statistic_.count(),
          Dtype(1) - bn_momentum_, batch_statistic_.cpu_data(),
          bn_momentum_, this->blobs_[2]->mutable_cpu_data());
    }
  }
  // Broadcast the mean vector
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
      Dtype(1), batch_sum_multiplier_.cpu_data(), batch_statistic_.cpu_data(),
      Dtype(0), spatial_statistic_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
      height_ * width_, 1, Dtype(-1),
      spatial_statistic_.cpu_data(), spatial_sum_multiplier_.cpu_data(),
      Dtype(0), broadcast_buffer_.mutable_cpu_data());
  // Subtract
  caffe_add(broadcast_buffer_.count(), const_bottom_data,
      broadcast_buffer_.cpu_data(), top_data);

  // Variance normalization
  if (this->phase_ == TEST && moving_average_) {
    // Use the moving average mean
    caffe_copy(batch_statistic_.count(), this->blobs_[3]->cpu_data(),
        batch_statistic_.mutable_cpu_data());
  } else if (this->phase_ == TRAIN) {
    caffe_powx(broadcast_buffer_.count(), const_top_data, Dtype(2),
        broadcast_buffer_.mutable_cpu_data());
    caffe_cpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_,
        Dtype(1) / (height_ * width_), broadcast_buffer_.cpu_data(),
        spatial_sum_multiplier_.cpu_data(), Dtype(0),
        spatial_statistic_.mutable_cpu_data());
    caffe_cpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1) / num_,
        spatial_statistic_.cpu_data(), batch_sum_multiplier_.cpu_data(),
        Dtype(0), batch_statistic_.mutable_cpu_data());
    // Add to the moving average
    if (moving_average_) {
      caffe_cpu_axpby(batch_statistic_.count(),
          Dtype(1) - bn_momentum_, batch_statistic_.cpu_data(),
          bn_momentum_, this->blobs_[3]->mutable_cpu_data());
    }
  }
  // Add eps
  caffe_add_scalar(batch_statistic_.count(), bn_eps_,
      batch_statistic_.mutable_cpu_data());
  // Standard deviation
  caffe_powx(batch_statistic_.count(), batch_statistic_.cpu_data(),
      Dtype(0.5), batch_statistic_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
      Dtype(1), batch_sum_multiplier_.cpu_data(), batch_statistic_.cpu_data(),
      Dtype(0), spatial_statistic_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
      height_ * width_, 1, Dtype(1),
      spatial_statistic_.cpu_data(), spatial_sum_multiplier_.cpu_data(),
      Dtype(0), broadcast_buffer_.mutable_cpu_data());
  // Divide by the std
  caffe_div(broadcast_buffer_.count(), const_top_data,
      broadcast_buffer_.cpu_data(), top_data);

  // Save the normalized inputs and std for backprop
  caffe_copy(broadcast_buffer_.count(), const_top_data,
      x_norm_.mutable_cpu_data());
  caffe_copy(batch_statistic_.count(), batch_statistic_.cpu_data(),
      x_std_.mutable_cpu_data());

  // Scale
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
      Dtype(1), batch_sum_multiplier_.cpu_data(), scale_data,
      Dtype(0), spatial_statistic_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
      height_ * width_, 1, Dtype(1),
      spatial_statistic_.cpu_data(), spatial_sum_multiplier_.cpu_data(),
      Dtype(0), broadcast_buffer_.mutable_cpu_data());
  caffe_mul(broadcast_buffer_.count(), const_top_data,
      broadcast_buffer_.cpu_data(), top_data);

  // Shift
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
      Dtype(1), batch_sum_multiplier_.cpu_data(), shift_data,
      Dtype(0), spatial_statistic_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
      height_ * width_, 1, Dtype(1),
      spatial_statistic_.cpu_data(), spatial_sum_multiplier_.cpu_data(),
      Dtype(0), broadcast_buffer_.mutable_cpu_data());
  caffe_add(broadcast_buffer_.count(), const_top_data,
      broadcast_buffer_.cpu_data(), top_data);
}

template <typename Dtype>
void BNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* const_bottom_diff = bottom[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* const_top_diff = top[0]->cpu_diff();

  Dtype* scale_diff = this->blobs_[0]->mutable_cpu_diff();
  Dtype* shift_diff = this->blobs_[1]->mutable_cpu_diff();
  const Dtype* scale_data = this->blobs_[0]->cpu_data();

  // gradient w.r.t. slope
  caffe_mul(broadcast_buffer_.count(), x_norm_.cpu_data(), const_top_diff,
      broadcast_buffer_.mutable_cpu_data());
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_,
      Dtype(1), broadcast_buffer_.cpu_data(),
      spatial_sum_multiplier_.cpu_data(), Dtype(0),
      spatial_statistic_.mutable_cpu_data());
  caffe_cpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1),
      spatial_statistic_.cpu_data(), batch_sum_multiplier_.cpu_data(),
      Dtype(1), scale_diff);

  // gradient w.r.t. bias
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_,
      Dtype(1), const_top_diff, spatial_sum_multiplier_.cpu_data(),
      Dtype(0), spatial_statistic_.mutable_cpu_data());
  caffe_cpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1),
      spatial_statistic_.cpu_data(), batch_sum_multiplier_.cpu_data(),
      Dtype(1), shift_diff);

  // gradient w.r.t. normalized inputs
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
      Dtype(1), batch_sum_multiplier_.cpu_data(), scale_data,
      Dtype(0), spatial_statistic_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
      height_ * width_, 1, Dtype(1), spatial_statistic_.cpu_data(),
      spatial_sum_multiplier_.cpu_data(), Dtype(0),
      broadcast_buffer_.mutable_cpu_data());
  caffe_mul(broadcast_buffer_.count(), const_top_diff,
      broadcast_buffer_.cpu_data(), broadcast_buffer_.mutable_cpu_data());

  // sum of x_hat * (dl / dx_hat)
  caffe_mul(broadcast_buffer_.count(), x_norm_.cpu_data(),
      broadcast_buffer_.cpu_data(), bottom_diff);
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_,
      Dtype(1), const_bottom_diff, spatial_sum_multiplier_.cpu_data(),
      Dtype(0), spatial_statistic_.mutable_cpu_data());
  caffe_cpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1),
      spatial_statistic_.cpu_data(), batch_sum_multiplier_.cpu_data(),
      Dtype(0), batch_statistic_.mutable_cpu_data());

  // x_hat times the sum
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
      Dtype(1), batch_sum_multiplier_.cpu_data(), batch_statistic_.cpu_data(),
      Dtype(0), spatial_statistic_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
      height_ * width_, 1, Dtype(1),
      spatial_statistic_.cpu_data(), spatial_sum_multiplier_.cpu_data(),
      Dtype(0), bottom_diff);
  caffe_mul(broadcast_buffer_.count(), x_norm_.cpu_data(),
      const_bottom_diff, bottom_diff);

  // Subtract the average of x_hat times the sum
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_,
      Dtype(1), broadcast_buffer_.cpu_data(),
      spatial_sum_multiplier_.cpu_data(), Dtype(0),
      spatial_statistic_.mutable_cpu_data());
  caffe_cpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1),
      spatial_statistic_.cpu_data(), batch_sum_multiplier_.cpu_data(),
      Dtype(0), batch_statistic_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
      Dtype(1), batch_sum_multiplier_.cpu_data(), batch_statistic_.cpu_data(),
      Dtype(0), spatial_statistic_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
      height_ * width_, 1, Dtype(1),
      spatial_statistic_.cpu_data(), spatial_sum_multiplier_.cpu_data(),
      Dtype(1), bottom_diff);
  caffe_cpu_axpby(broadcast_buffer_.count(), Dtype(1),
      broadcast_buffer_.cpu_data(), Dtype(-1) / (num_ * height_ * width_),
      bottom_diff);

  // Divide by the std
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
      Dtype(1), batch_sum_multiplier_.cpu_data(), x_std_.cpu_data(),
      Dtype(0), spatial_statistic_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
      height_ * width_, 1, Dtype(1),
      spatial_statistic_.cpu_data(), spatial_sum_multiplier_.cpu_data(),
      Dtype(0), broadcast_buffer_.mutable_cpu_data());
  caffe_div(broadcast_buffer_.count(), const_bottom_diff,
      broadcast_buffer_.cpu_data(), bottom_diff);
}


#ifdef CPU_ONLY
STUB_GPU(BNLayer);
#endif

INSTANTIATE_CLASS(BNLayer);
REGISTER_LAYER_CLASS(BN);

}  // namespace caffe