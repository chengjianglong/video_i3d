#include <stdio.h>  // for snprintf
// #include <cuda_runtime.h>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/image_io.hpp"
#include "caffe/util/io.hpp"
#include "google/protobuf/text_format.h"

using namespace caffe;  // NOLINT(build/namespaces)

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv);

int main(int argc, char** argv) {
  return feature_extraction_pipeline<float>(argc, argv);
}

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv) {
  char* net_proto = argv[1];
  char* pretrained_model = argv[2];
  int device_id = atoi(argv[3]);
  uint batch_size = atoi(argv[4]);
  int FLAG_iterations = atoi(argv[5]);
  std::string resfname = std::string(argv[6]);
  std::ofstream resfile;
  resfile.open(resfname.c_str());


  //Caffe::set_phase(Caffe::TEST);
  Caffe::set_mode(Caffe::GPU);
  /*
  if (device_id >= 0) {
      Caffe::set_mode(Caffe::GPU);
      Caffe::SetDevice(device_id);
      LOG(ERROR) << "Using GPU #" << device_id;
  } else {
      Caffe::set_mode(Caffe::CPU);
      LOG(ERROR) << "Using CPU";
  }
  */

  shared_ptr<Net<Dtype> > feature_extraction_net(
      new Net<Dtype>(string(net_proto), caffe::TEST));
  feature_extraction_net->CopyTrainedLayersFrom(string(pretrained_model));

  for (int i = 7; i < argc; i++) {
  CHECK(feature_extraction_net->has_blob(string(argv[i])))
      << "Unknown feature blob name " << string(argv[i])
      << " in the network " << string(net_proto);
  }

  float accuracy = 0;
  float loss = 0;
  for(int i=0; i < FLAG_iterations; i++)
  {
	  float iter_loss;
	  const vector<Blob<float>*>& result = feature_extraction_net->Forward(&iter_loss);

	  const boost::shared_ptr<Blob<float> > blob_feat_fc = feature_extraction_net->blob_by_name("fc8");
	  int batch_size = blob_feat_fc->num();
	  int dim_features = blob_feat_fc->count()/batch_size;
	  const float *data_blob_feat_fc;

	  std::cout << "batch_size : " << batch_size << std::endl;

	  resfile << i;
	  for(int n=0; n<batch_size; ++n)
	  {
		  data_blob_feat_fc = blob_feat_fc->cpu_data() + blob_feat_fc->offset(n);

		  for(int d=0; d<dim_features; ++d)
		  {
			  std::cout << "fc8[" << d << "] = " << data_blob_feat_fc[d] << std::endl;
			  resfile << " " << data_blob_feat_fc[d];
		  }
	  }

	  const boost::shared_ptr<Blob<float> > blob_label = feature_extraction_net->blob_by_name("label");
	  int label_size = blob_label->num();
	  int dim_labels = blob_label->count()/label_size;
	  const float *data_blob_label;

	  //std::cout << "label_size : " << label_size << std::endl;

	  for(int n=0; n<label_size; ++n)
	  {
		  data_blob_label = blob_label->cpu_data() + blob_label->offset(n);

		  for(int d=0; d<dim_labels; ++d)
		  {
			  std::cout << "label[" << d << "] = " << data_blob_label[d] << std::endl;
		      resfile << " " << data_blob_label[d];
		  }
	  }
	  
	  resfile << std::endl;

	  accuracy += result[0]->cpu_data()[0];
	  loss += iter_loss;
	  int idx = 0;
	  for(int j=0; j<result.size(); j++)
	  {
		  const float* result_vec = result[j]->cpu_data();
		  for(int k=0; k<result[j]->count(); ++k, ++idx)
		  {
			  const float score = result_vec[k];
			  const std::string& output_name = feature_extraction_net->blob_names()[feature_extraction_net->output_blob_indices()[j]];
			  LOG(INFO) << "Batch " << i << " , " << output_name << " = " << score;
		  }
	  }
  }

  resfile.close();

  accuracy /= FLAG_iterations;
  loss /= FLAG_iterations;
  LOG(INFO) << "Accuracy: " << accuracy;
  LOG(INFO) << "Loss: " << loss;

  return 0;
}

