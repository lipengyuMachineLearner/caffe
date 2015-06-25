// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>
#include <fstream>
#include  <io.h>
#include <direct.h>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"
#include "opencvlib.h"

using std::max;

namespace caffe {

const float kLOG_THRESHOLD = 1e-20;

template <typename Dtype>
void MultinomialLogisticLossLayer<Dtype>::SetUp(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 2) << "Loss Layer takes two blobs as input.";
  CHECK_EQ(top->size(), 0) << "Loss Layer takes no output.";
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
}

template <typename Dtype>
Dtype MultinomialLogisticLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  Dtype loss = 0;
  for (int i = 0; i < num; ++i) {
    int label = static_cast<int>(bottom_label[i]);
    Dtype prob = max(bottom_data[i * dim + label], Dtype(kLOG_THRESHOLD));
    loss -= log(prob);
  }
  return loss / num;
}

template <typename Dtype>
void MultinomialLogisticLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* bottom_data = (*bottom)[0]->cpu_data();
  const Dtype* bottom_label = (*bottom)[1]->cpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  int num = (*bottom)[0]->num();
  int dim = (*bottom)[0]->count() / (*bottom)[0]->num();
  memset(bottom_diff, 0, sizeof(Dtype) * (*bottom)[0]->count());
  for (int i = 0; i < num; ++i) {
    int label = static_cast<int>(bottom_label[i]);
    Dtype prob = max(bottom_data[i * dim + label], Dtype(kLOG_THRESHOLD));
    bottom_diff[i * dim + label] = -1. / prob / num;
  }
}


template <typename Dtype>
void InfogainLossLayer<Dtype>::SetUp(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 2) << "Loss Layer takes two blobs as input.";
  CHECK_EQ(top->size(), 0) << "Loss Layer takes no output.";
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  BlobProto blob_proto;
  ReadProtoFromBinaryFile(this->layer_param_.infogain_loss_param().source(),
                          &blob_proto);
  infogain_.FromProto(blob_proto);
  CHECK_EQ(infogain_.num(), 1);
  CHECK_EQ(infogain_.channels(), 1);
  CHECK_EQ(infogain_.height(), infogain_.width());
}


template <typename Dtype>
Dtype InfogainLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const Dtype* infogain_mat = infogain_.cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  CHECK_EQ(infogain_.height(), dim);
  Dtype loss = 0;
  for (int i = 0; i < num; ++i) {
    int label = static_cast<int>(bottom_label[i]);
    for (int j = 0; j < dim; ++j) {
      Dtype prob = max(bottom_data[i * dim + j], Dtype(kLOG_THRESHOLD));
      loss -= infogain_mat[label * dim + j] * log(prob);
    }
  }
  return loss / num;
}

template <typename Dtype>
void InfogainLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* bottom_data = (*bottom)[0]->cpu_data();
  const Dtype* bottom_label = (*bottom)[1]->cpu_data();
  const Dtype* infogain_mat = infogain_.cpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  int num = (*bottom)[0]->num();
  int dim = (*bottom)[0]->count() / (*bottom)[0]->num();
  CHECK_EQ(infogain_.height(), dim);
  for (int i = 0; i < num; ++i) {
    int label = static_cast<int>(bottom_label[i]);
    for (int j = 0; j < dim; ++j) {
      Dtype prob = max(bottom_data[i * dim + j], Dtype(kLOG_THRESHOLD));
      bottom_diff[i * dim + j] = - infogain_mat[label * dim + j] / prob / num;
    }
  }
}


template <typename Dtype>
void EuclideanLossLayer<Dtype>::SetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 2) << "Loss Layer takes two blobs as input.";
  CHECK_EQ(top->size(), 0) << "Loss Layer takes no as output.";
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  difference_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
Dtype EuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  caffe_sub(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(),
      difference_.mutable_cpu_data());
  Dtype loss = caffe_cpu_dot(
      count, difference_.cpu_data(), difference_.cpu_data()) / num / Dtype(2);
  return loss;
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  int count = (*bottom)[0]->count();
  int num = (*bottom)[0]->num();
  // Compute the gradient
  caffe_cpu_axpby(count, Dtype(1) / num, difference_.cpu_data(), Dtype(0),
      (*bottom)[0]->mutable_cpu_diff());
}

template <typename Dtype>
void AccuracyLayer<Dtype>::SetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 2) << "Accuracy Layer takes two blobs as input.";
  CHECK_EQ(top->size(), 1) << "Accuracy Layer takes 1 output.";
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  (*top)[0]->Reshape(1, 2, 1, 1);
}

template <typename Dtype>
Dtype AccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  Dtype accuracy = 0;
  Dtype logprob = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  for (int i = 0; i < num; ++i) {
    // Accuracy
    Dtype maxval = -FLT_MAX;
    int max_id = 0;
    for (int j = 0; j < dim; ++j) {
      if (bottom_data[i * dim + j] > maxval) {
        maxval = bottom_data[i * dim + j];
        max_id = j;
      }
    }
    if (max_id == static_cast<int>(bottom_label[i])) {
      ++accuracy;
    }
    Dtype prob = max(bottom_data[i * dim + static_cast<int>(bottom_label[i])],
                     Dtype(kLOG_THRESHOLD));
    logprob -= log(prob);
  }
  // LOG(INFO) << "Accuracy: " << accuracy;
  (*top)[0]->mutable_cpu_data()[0] = accuracy / num;
  (*top)[0]->mutable_cpu_data()[1] = logprob / num;
  // Accuracy layer should not be used as a loss function.
  return Dtype(0);
}

/**********************li pengyu add start**************************************/
template <typename Dtype>
void OutAccuracyLayer<Dtype>::SetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 3) << "Accuracy Layer takes three blobs as input.";
  CHECK_EQ(top->size(), 1) << "Accuracy Layer takes 1 output.";
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_EQ(bottom[0]->num(), bottom[2]->num())
      << "The data and image should have the same number.";
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  (*top)[0]->Reshape(1, 2, 1, 1);

  index = 1;
  outFile.open(".//ClassfierResult.txt");
}

template <typename Dtype>
Dtype OutAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {

  Dtype accuracy = 0;
  Dtype logprob = 0;
  
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();

  int width = bottom[2]->width();
  int height = bottom[2]->height();
  int channels = bottom[2]->channels();
  int step = width*channels;
  int numStep = step*height;

  for (int i = 0; i < num; ++i) {
    // Accuracy
    Dtype maxval = -FLT_MAX;
    int max_id = 0;

	outFile << index << " ";
	
    for (int j = 0; j < dim; ++j) {
	
      if (bottom_data[i * dim + j] > maxval) {
        maxval = bottom_data[i * dim + j];
        max_id = j;
      }

	  outFile << bottom_data[i * dim + j] << " "; 
    }
    if (max_id == static_cast<int>(bottom_label[i])) {
      ++accuracy;
    }
	else
	{
		channels = channels < 3 ? channels : 3;
		IplImage *img = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, channels);
		const Dtype *dataPoint = bottom[2]->cpu_data();
		unsigned char *imgData = (unsigned char *)img->imageData;
		for(int h = 0 ; h < height ; h++)
			for(int w = 0 ; w < width ; w++)
				for(int c = 0 ; c < channels ; c++)
					imgData[c + w*channels + h*width*channels] = dataPoint[w + h*width + c*width*height + i*numStep]*255;
		
		

		string path = ".//Errorimg";
		
		if( (_access( path.c_str(), 0 )) == -1 )
		{
			_mkdir(path.c_str());
		}
		
		char tmp[10];
		sprintf(tmp, "%d", static_cast<int>(bottom_label[i]));
		path = path + "//" +tmp;
		if( (_access( path.c_str(), 0 )) == -1 )
		{
			_mkdir(path.c_str());
		}

		sprintf(tmp, "%d", index);
		string fileName = path + "//" + tmp;
		sprintf(tmp, "%d", max_id);
		fileName = fileName + "_" + tmp + ".png";

		cvSaveImage(fileName.c_str(), img );
		cvReleaseImage(&img);
	}
    Dtype prob = max(bottom_data[i * dim + static_cast<int>(bottom_label[i])],
                     Dtype(kLOG_THRESHOLD));
    logprob -= log(prob);

	outFile << bottom_data[i * dim + max_id] << " " << bottom_data[i * dim + static_cast<int>(bottom_label[i])] << " ";
	outFile << max_id << " " << static_cast<int>(bottom_label[i]) << endl;

	index++;
  }
  // LOG(INFO) << "Accuracy: " << accuracy;
  (*top)[0]->mutable_cpu_data()[0] = accuracy / num;
  (*top)[0]->mutable_cpu_data()[1] = logprob / num;

  
  // Accuracy layer should not be used as a loss function.


  return Dtype(0);
}

template <typename Dtype>
void OutPreLayerInfoLayer<Dtype>::SetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size() , 2) << "OutPreLayerInfoLayer takes should be two";
  CHECK_LE(top->size() , 1) << "OutPreLayerInfoLayer takes more than one blobs as output";
  string path = this->layer_param_.outprelayer_param().datafile();
  signFirst = true;

  outFile.open(path);

  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  (*top)[0]->Reshape(num_, channels_, height_, width_);

}

template <typename Dtype>
Dtype OutPreLayerInfoLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {


  int count = bottom[0]->count();
  const Dtype *bottom_data = bottom[0]->cpu_data();
  const Dtype *bottom_label = bottom[1]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  caffe_copy(count, bottom_data, top_data);




  int step = width_*channels_;
  int numStep = step*height_;

  if(signFirst)
  {
	  outFile << channels_ << "," << height_ << "," << width_ << "," << endl;
	  signFirst = false;
  }


  for(int n = 0 ; n < num_ ; n++)
  {
	  outFile << static_cast<int>(bottom_label[n]) << ",";
	  for(int c = 0 ; c < channels_ ; c++)
	  {
		  for(int h = 0 ; h < height_; h++)
		  {
			  for(int w = 0 ; w < width_ ; w++)
			  {
				  outFile << bottom_data[w + h*width_ + c*width_*height_ + n*numStep] << ",";
			  }
		  }
	  }
	  outFile << endl;
  }

  return Dtype(0);
}

template <typename Dtype>
void SubClassMapLayer<Dtype>::SetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {

	

  CHECK_EQ(bottom.size(), 1) << "Subclass Layer takes one blobs as input.";
  CHECK_EQ(top->size(), 1) << "Subclass Layer takes 1 output.";

  type_ = this->layer_param_.subclasslayer_param().subclass_type();
  CHECK_LE(type_, 1) << "the type of subclass should be <= 1 " ;
  switch(type_)
  {
  case 0 :
	  {//0-���֣�1-ƽ�棬2-��ĸ��3-ͼ����4-��ͷ��5-��ֹƽ�棬6 ��ת��ͷ
			subclass[0] = 0;
			subclass[1] = 0;
			subclass[2] = 0;
			subclass[3] = 0;
			subclass[4] = 0;
			subclass[5] = 0;
			subclass[6] = 0;
			subclass[7] = 0;
			subclass[8] = 0;
			subclass[9] = 3;
			subclass[10] = 3;
			subclass[11] = 3;
			subclass[12] = 1;
			subclass[13] = 1;
			subclass[14] = 2;
			subclass[15] = 1;
			subclass[16] = 3;
			subclass[17] = 3;
			subclass[18] = 3;
			subclass[19] = 4;
			subclass[20] = 4;
			subclass[21] = 4;
			subclass[22] = 3;
			subclass[23] = 3;
			subclass[24] = 3;
			subclass[25] = 3;
			subclass[26] = 3;
			subclass[27] = 3;
			subclass[28] = 3;
			subclass[29] = 3;
			subclass[30] = 3;
			subclass[31] = 3;
			subclass[32] = 5;
			subclass[33] = 4;
			subclass[34] = 4;
			subclass[35] = 4;
			subclass[36] = 4;
			subclass[37] = 4;
			subclass[38] = 4;
			subclass[39] = 4;
			subclass[40] = 6;
			subclass[41] = 3;
			subclass[42] = 3;

			break;
	  }
  case 1:
	  {// 0-�ڰ� 1-��� 2-�� 3-�� 4-��� 5-��ɫ
			subclass[0] = 1;
			subclass[1] = 1;
			subclass[2] = 1;
			subclass[3] = 1;
			subclass[4] = 1;
			subclass[5] = 1;
			subclass[6] = 0;
			subclass[7] = 1;
			subclass[8] = 1;
			subclass[9] = 1;
			subclass[10] = 1;
			subclass[11] = 1;
			subclass[12] = 2;
			subclass[13] = 4;
			subclass[14] = 4;
			subclass[15] = 4;
			subclass[16] = 1;
			subclass[17] = 4;
			subclass[18] = 1;
			subclass[19] = 1;
			subclass[20] = 1;
			subclass[21] = 1;
			subclass[22] = 1;
			subclass[23] = 1;
			subclass[24] = 1;
			subclass[25] = 1;
			subclass[26] = 5;
			subclass[27] = 1;
			subclass[28] = 1;
			subclass[29] = 1;
			subclass[30] = 1;
			subclass[31] = 1;
			subclass[32] = 0;
			subclass[33] = 3;
			subclass[34] = 3;
			subclass[35] = 3;
			subclass[36] = 3;
			subclass[37] = 3;
			subclass[38] = 3;
			subclass[39] = 3;
			subclass[40] = 3;
			subclass[41] = 0;
			subclass[42] = 0;

			break;
	  }
  default:
	  {
		 LOG(FATAL) << "the type of subclass should be <= 1 ";
		 return;
	  }
  }

  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  (*top)[0]->Reshape(num_, channels_, height_, width_);
	
}

template <typename Dtype>
Dtype SubClassMapLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  Dtype accuracy = 0;
  Dtype logprob = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();

  int step = width_*channels_;
  int numStep = step*height_;
  for(int n = 0 ; n < num_ ; n++)
  {
	  for(int c = 0 ; c < channels_ ; c++)
	  {
		  for(int h = 0 ; h < height_; h++)
		  {
			  for(int w = 0 ; w < width_ ; w++)
			  {
				  top_data[w + h*width_ + c*width_*height_ + n*numStep] = subclass[static_cast<int>(bottom_data[w + h*width_ + c*width_*height_ + n*numStep])];
			  }
		  }
	  }
  }
  return Dtype(0);
}

/**********************li pengyu add over**************************************/


template <typename Dtype>
void HingeLossLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 2) << "Hinge Loss Layer takes two blobs as input.";
  CHECK_EQ(top->size(), 0) << "Hinge Loss Layer takes no output.";
}

template <typename Dtype>
Dtype HingeLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;

  caffe_copy(count, bottom_data, bottom_diff);
  for (int i = 0; i < num; ++i) {
    bottom_diff[i * dim + static_cast<int>(label[i])] *= -1;
  }
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < dim; ++j) {
      bottom_diff[i * dim + j] = max(Dtype(0), 1 + bottom_diff[i * dim + j]);
    }
  }
  return caffe_cpu_asum(count, bottom_diff) / num;
}

template <typename Dtype>
void HingeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  const Dtype* label = (*bottom)[1]->cpu_data();
  int num = (*bottom)[0]->num();
  int count = (*bottom)[0]->count();
  int dim = count / num;

  caffe_cpu_sign(count, bottom_diff, bottom_diff);
  for (int i = 0; i < num; ++i) {
    bottom_diff[i * dim + static_cast<int>(label[i])] *= -1;
  }
  caffe_scal(count, Dtype(1. / num), bottom_diff);
}

INSTANTIATE_CLASS(MultinomialLogisticLossLayer);
INSTANTIATE_CLASS(InfogainLossLayer);
INSTANTIATE_CLASS(EuclideanLossLayer);
INSTANTIATE_CLASS(AccuracyLayer);
INSTANTIATE_CLASS(OutAccuracyLayer);
INSTANTIATE_CLASS(HingeLossLayer);
INSTANTIATE_CLASS(OutPreLayerInfoLayer);
INSTANTIATE_CLASS(SubClassMapLayer);
}  // namespace caffe
