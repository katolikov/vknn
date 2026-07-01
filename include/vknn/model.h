// High-level API. Load a model and run it — no tensor names, shapes, or dtypes to wire up by hand;
// everything is read from the model. The lower-level Session / IOTensor in session.h remain
// available for advanced control.
//
//   vknn::Model net = vknn::Model::load("mobilenet.onnx");
//   vknn::Tensor out = net.run(pixels);   // pixels: std::vector<float>, NCHW
//   int cls = out.argmax();
#pragma once
#include "vknn/config.h"
#include "vknn/dtype.h"
#include "vknn/tensor_format.h"

#include "vknn/tensor_info.h"  // struct TensorInfo
#include "vknn/tensor_class.h" // class Tensor
#include "vknn/model_class.h"  // class Model (+ forward decl class Session)
#include "vknn/find_tensor.h"  // findTensor()
