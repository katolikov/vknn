// The friendly API in action: load a model, run it, read the result — no tensor wiring.
//   vx_predict model.onnx input.bin
#include <cstdio>
#include <fstream>
#include <vector>

#include "vknn/model.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("usage: %s model.onnx [input.bin]\n", argv[0]);
    return 1;
  }
  vknn::Model net = vknn::Model::load(argv[1]);  // precision auto, Vulkan if available
  if (!net) {
    printf("failed to load %s\n", argv[1]);
    return 1;
  }

  // Discover what the model wants — purely informational; you never have to set it.
  for (auto& in : net.inputs())
    printf("input  '%s'  %s  (%lld values)\n", in.name.c_str(), in.shapeString().c_str(),
           (long long)in.count);
  for (auto& out : net.outputs())
    printf("output '%s'  %s\n", out.name.c_str(), out.shapeString().c_str());

  // Build the input from a raw fp32 file (or zeros). The model knows the shape.
  int64_t need = net.inputs().empty() ? 0 : net.inputs()[0].count;
  std::vector<float> input(need, 0.f);
  if (argc >= 3) {
    std::ifstream f(argv[2], std::ios::binary);
    if (f)
      f.read(reinterpret_cast<char*>(input.data()), need * sizeof(float));
  }

  vknn::Tensor out = net.run(input);  // one line to run
  printf("result: shape=%s  top1=%lld  max=%.4f\n", out.shapeString().c_str(),
         (long long)out.argmax(), out.max());

  if (argc >= 4) {  // optional: save the optimized model for fast reloads
    if (net.save(argv[3]))
      printf("saved optimized model -> %s\n", argv[3]);
  }
  return 0;
}
