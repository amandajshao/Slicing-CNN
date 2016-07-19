// This program converts a set of labels to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_labelset [FLAGS] LISTFILE DB_NAME
//
// where LISTFILE should be a list of labels, in the format as
//   7
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
using std::string;
using boost::scoped_ptr;

DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of labels");
DEFINE_string(backend, "lmdb",
        "The backend {lmdb, leveldb} for storing the result");

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of labels to the leveldb/lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_labelset [FLAGS] LISTFILE DB_NAME\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 3) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_labelset");
    return 1;
  }

  std::ifstream infile(argv[1]);
  std::vector<int> lines;
  int label;
  while (infile >> label) {
    lines.push_back(label);
  }
  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(lines.begin(), lines.end());
  }
  LOG(INFO) << "A total of " << lines.size() << " labels.";

  // Create new DB
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(argv[2], db::NEW);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());

  // Storing to db
  Datum datum;
  datum.set_channels(1);
  datum.set_height(1);
  datum.set_width(1);
  datum.clear_data();
  int count = 0;
  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];

  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    datum.clear_float_data();
    datum.add_float_data(lines[line_id]);

    // sequential
    int length = snprintf(key_cstr, kMaxKeyLength, "%08d", line_id);

    // Put in db
    string out;
    CHECK(datum.SerializeToString(&out));
    txn->Put(string(key_cstr, length), out);

    if (++count % 1000 == 0) {
        // Commit db
        txn->Commit();
        txn.reset(db->NewTransaction());
        LOG(ERROR) << "Processed " << count << " files.";
      }
    }
  // write the last batch
  if (count % 1000 != 0) {
    txn->Commit();
    LOG(ERROR) << "Processed " << count << " files.";
  }
  return 0;
}