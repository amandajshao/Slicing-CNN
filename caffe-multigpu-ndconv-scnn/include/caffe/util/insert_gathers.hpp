#ifdef USE_MPI
#ifndef INSERT_GATHERS_HPP_
#define INSERT_GATHERS_HPP_

#include <algorithm>
#include <set>
#include <string>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

void InsertGathers(const set<string>& serial_layers, NetParameter* param);

void ConfigureGatherLayer(const string& layer_down_name,
                          const string& layer_up_name,
                          const vector<pair<int, string> >& connecting_blobs,
                          LayerParameter* layer_param);

}

#endif // INSERT_GATHERS_HPP_
#endif // USE_MPI