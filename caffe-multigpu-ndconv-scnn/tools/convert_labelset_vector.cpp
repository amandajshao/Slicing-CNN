// This program converts a set of labels to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_labelset [FLAGS] LISTFILE DB_NAME
//
// where LISTFILE is a list of keys with corresponding labels,
// the first line is a number indicating the length of labels
//
// 10
// key1 1 2 3 4 5 6 7 8 9 10
// key2 10 9 8 7 6 5 4 3 2 1
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_string(backend, "lmdb",
        "The backend {lmdb, leveldb} for storing the result");


int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images to the leveldb/lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_labelset LISTFILE DB_NAME\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 2) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_labelset");
    return 1;
  }

  std::ifstream infile(argv[1]);
  int num_label;
  infile >> num_label;
  std::string key;

  // Create new DB
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(argv[2], db::NEW);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());

  int count = 0;
  while (infile >> key) {
    Datum datum;
    datum.set_channels(num_label);
    datum.set_height(1);
    datum.set_width(1);
    for (int i = 0; i < num_label; ++i) {
      float label;
      infile >> label;
      datum.add_float_data(label);
    }
    // Put in db
    string out;
    CHECK(datum.SerializeToString(&out));
    txn->Put(key, out);
    if (++count % 1000 == 0) {
      // Commit db
      txn->Commit();
      txn.reset(db->NewTransaction());
      LOG(ERROR) << "Processed " << count << " samples.";
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    txn->Commit();
    LOG(ERROR) << "Processed " << count << " files.";
  }
  LOG(INFO) << "A total of " << count << " images.";
  return 0;
}
