#ifdef USE_MPI
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/mpi_templates.hpp"

namespace caffe {

template <typename Dtype>
void MPIGatherLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), top.size())
      << "The number of bottom and top blobs must be the same";
}

template <typename Dtype>
void MPIGatherLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < bottom.size(); ++i) {
    vector<int> shape = bottom[i]->shape();
    shape[0] *= Caffe::mpi_size();
    top[i]->Reshape(shape);
  }
}

template <typename Dtype>
void MPIGatherLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < bottom.size(); ++i) {
    MPIGather<Dtype>(bottom[i]->count(), bottom[i]->cpu_data(),
        top[i]->mutable_cpu_data());
  }
}

template <typename Dtype>
void MPIGatherLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < bottom.size(); ++i) {
    MPIScatter<Dtype>(bottom[i]->count(), top[i]->cpu_diff(),
        bottom[i]->mutable_cpu_diff());
  }
}

INSTANTIATE_CLASS(MPIGatherLayer);
REGISTER_LAYER_CLASS(MPIGather);

} // namespace caffe

#endif // USE_MPI