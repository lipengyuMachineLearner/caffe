// Copyright 2014 BVLC and contributors.
// This program converts a set of images to a leveldb by storing them as Datum
// proto buffers.
// Usage:
//    convert_imageset ROOTFOLDER/ LISTFILE DB_NAME [0/1]
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....
// if the last argument is 1, a random shuffle will be carried out before we
// process the file lines.

#include <glog/logging.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using std::string;

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc < 4 || argc > 5) {
    printf("Convert a set of images to the leveldb format used\n"
        "as input for Caffe.\n"
        "Usage:\n"
        "    convert_imageset ROOTFOLDER/ LISTFILE DB_NAME"
        " RANDOM_SHUFFLE_DATA[0 or 1]\n"
        "The ImageNet dataset for the training demo is at\n"
        "    http://www.image-net.org/download-images\n");
    return 1;
  }
  std::ifstream infile(argv[2]);
  if(!infile)
	  LOG(INFO) <<"there is no file named " << argv[2];
  std::vector<string> lines;
  string infor;
  int label;
  while (infile >> infor) {
    lines.push_back(infor);
  }
  if (argc == 5 && argv[4][0] == '1') {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    std::random_shuffle(lines.begin()+1, lines.end());
  }
  LOG(INFO) << "A total of " << lines.size() << " images.";

  leveldb::DB* db;
  leveldb::Options options;
  options.error_if_exists = true;
  options.create_if_missing = true;
  options.write_buffer_size = 268435456;
  LOG(INFO) << "Opening leveldb " << argv[3];
  leveldb::Status status = leveldb::DB::Open(
      options, argv[3], &db);
  CHECK(status.ok()) << "Failed to open leveldb " << argv[3];

  Datum datum;
  int count = 0;
  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];
  leveldb::WriteBatch* batch = new leveldb::WriteBatch();
  int data_size;
  bool data_size_initialized = false;

  int width = 0 , height = 0 , channel = 0;
  std::string::size_type pos1 = 0 , pos2 = 0;
  pos2 = lines[0].find(",", pos1);
  channel = atoi(lines[0].substr(pos1,pos2-pos1).c_str());

  pos1 = pos2 + 1;
  pos2 = lines[0].find(",", pos1);
  height = atoi(lines[0].substr(pos1,pos2-pos1).c_str());

  pos1 = pos2 + 1;
  pos2 = lines[0].find(",", pos1);
  width = atoi(lines[0].substr(pos1,pos2-pos1).c_str());

  for (int line_id = 1; line_id < lines.size(); ++line_id) {
    if (!ReadCSVToDatum(lines[line_id], channel, width, height, &datum)) {
      continue;
    }

    if (!data_size_initialized) {
      data_size = datum.channels() * datum.height() * datum.width();
      data_size_initialized = true;
    } else {
      ::google::protobuf::RepeatedField< float > data = datum.float_data();
      CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
          << data.size();
    }
    // sequential
    snprintf(key_cstr, kMaxKeyLength, "%08d_%s", line_id,
        lines[line_id].c_str());
    string value;
    // get the value
    datum.SerializeToString(&value);
    batch->Put(string(key_cstr), value);
    if (++count % 1000 == 0) {
      db->Write(leveldb::WriteOptions(), batch);
      LOG(ERROR) << "Processed " << count << " files.";
      delete batch;
      batch = new leveldb::WriteBatch();
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    db->Write(leveldb::WriteOptions(), batch);
    LOG(ERROR) << "Processed " << count << " files.";
  }

  delete batch;
  delete db;
  return 0;
}
