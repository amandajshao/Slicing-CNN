#ifdef USE_MPI
#include <utility>

#include "caffe/util/insert_gathers.hpp"

using std::cout;
using std::endl;

namespace caffe {

void InsertGathers(const set<string>& serial_layers, NetParameter* param) {
  const int num_layers = param->layer_size();
  // Build the DAG
  vector<set<string> > layer_tops(num_layers);
  for (int i = 0; i < num_layers; ++i) {
    const LayerParameter& layer_param = param->layer(i);
    for (int j = 0; j < layer_param.top_size(); ++j) {
      layer_tops[i].insert(layer_param.top(j));
    }
  }
  // Insert MPIGatherLayer between parallel --> serial layers
  NetParameter param_gather;
  param_gather.CopyFrom(*param);
  param_gather.clear_layer();
  for (int i = 0; i < num_layers; ++i) {
    const LayerParameter& layer_up = param->layer(i);
    if (serial_layers.find(layer_up.name()) == serial_layers.end() ||
        layer_up.type() == "MPIGather" ||
	layer_up.type() == "SoftmaxWithLoss" ||
	layer_up.type() == "SigmoidCrossEntropyLoss" ||
	layer_up.type() == "Accuracy") {
      LayerParameter* layer = param_gather.add_layer();
      layer->CopyFrom(layer_up);
      continue;
    }
    // First insert all the needed gather layers, then the upper layer, to make
    // all the layers topo-ordered.
    vector<pair<int, string> > update_bottoms;
    // There are some layers handling some blobs in-place. We use a reverse
    // for-loop to find the nearest layer.
    set<string> blobs_used;
    for (int j = i - 1; j >= 0; --j) {
      const LayerParameter& layer_down = param->layer(j);
      if (serial_layers.find(layer_down.name()) != serial_layers.end())
          continue;
      // Find the blobs connecting these two layers
      const set<string>& tops = layer_tops[j];
      vector<pair<int, string> > connecting_blobs;
      for (int k = 0; k < layer_up.bottom_size(); ++k) {
        const string& name = layer_up.bottom(k);
        if (blobs_used.find(name) != blobs_used.end()) continue;
        if (tops.find(name) != tops.end()) {
          connecting_blobs.push_back(make_pair(k, name));
          blobs_used.insert(name);
        }
      }
      if (connecting_blobs.empty()) continue;
      // Insert a MPIGatherLayer
      LayerParameter* gather_layer = param_gather.add_layer();
      ConfigureGatherLayer(layer_down.name(), layer_up.name(),
          connecting_blobs, gather_layer);
      // Record the bottom blobs of layer_up to be updated
      for (int k = 0; k < connecting_blobs.size(); ++k) {
        const int blob_idx = connecting_blobs[k].first;
        update_bottoms.push_back(make_pair(blob_idx, gather_layer->top(k)));
      }
    }
    // Insert the upper layer
    LayerParameter* layer = param_gather.add_layer();
    layer->CopyFrom(layer_up);
    for (int k = 0; k < update_bottoms.size(); ++k) {
      layer->set_bottom(update_bottoms[k].first, update_bottoms[k].second);
    }
  }
  param->CopyFrom(param_gather);
}

void ConfigureGatherLayer(const string& layer_down_name,
                          const string& layer_up_name,
                          const vector<pair<int, string> >& connecting_blobs,
                          LayerParameter* layer_param) {
  layer_param->Clear();
  layer_param->set_type("MPIGather");
  layer_param->set_name("gather_" + layer_down_name + "_to_" + layer_up_name);
  for (int i = 0; i < connecting_blobs.size(); ++i) {
    const string& bottom_blob_name = connecting_blobs[i].second;
    layer_param->add_bottom(bottom_blob_name);
    layer_param->add_top("gathered_" + bottom_blob_name);
  }
}

} // namespace caffe

#endif // USE_MPI
