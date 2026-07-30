// convertInitializersFp16's coordinate-class keep rule: an initializer whose bits reach a
// GridSample coordinate operand through elementwise/movement algebra keeps its fp32 payload
// (its precision is the sample position), a spatial-axis ramp keeps it by shape, and an
// ordinary weight on the data path still narrows to fp16.
#include "import/passes.h"
#include "vknn/dtype.h"
#include <gtest/gtest.h>

using namespace vknn;

namespace {

    TensorId addAct(Graph &g, const std::string &name, Shape shape) {
        TensorDesc d;
        d.name  = name;
        d.shape = std::move(shape);
        return g.addTensor(d);
    }

    TensorId addFloatInit(Graph &g, const std::string &name, const Shape &shape) {
        TensorDesc d;
        d.name          = name;
        d.shape         = shape;
        d.isInitializer = true;
        TensorId      t = g.addTensor(d);
        HostBuffer    hb;
        const int64_t n = numElements(shape) > 0 ? numElements(shape) : 1;
        hb.resizeElems(n, DType::Float32);
        for (int64_t i = 0; i < n; ++i)
        {
            hb.f32()[i] = 0.25f * (float) (i + 1);
        }
        g.initializers[t] = hb;
        return t;
    }

    Node makeNode(OpType type, const std::string &name, std::vector<TensorId> inputs, std::vector<TensorId> outputs) {
        Node n;
        n.type    = type;
        n.name    = name;
        n.inputs  = std::move(inputs);
        n.outputs = std::move(outputs);
        return n;
    }

} // namespace

TEST(ConvertFp16Coord, GridChainKeepsFp32AndWeightNarrows) {
    Graph g;
    // image -> (Binary with a plain weight) -> sampled by GridSample whose grid comes from a
    // scalar-scaled constant grid: grid algebra = Binary(gridConst, zoomScalar).
    TensorId x         = addAct(g, "x", {1, 4, 8, 8});
    TensorId weight    = addFloatInit(g, "w_plain", {1, 4, 1, 1}); // data-path constant: must narrow
    TensorId scaled    = addAct(g, "scaled", {1, 4, 8, 8});
    TensorId gridConst = addFloatInit(g, "grid_c", {1, 8, 8, 2}); // coordinate constant: must keep fp32
    TensorId zoom      = addFloatInit(g, "zoom", {});             // scalar in the grid algebra: must keep fp32
    TensorId grid      = addAct(g, "grid", {1, 8, 8, 2});
    TensorId y         = addAct(g, "y", {1, 4, 8, 8});

    Node mulData = makeNode(OpType::Binary, "mul_data", {x, weight}, {scaled});
    Node mulGrid = makeNode(OpType::Binary, "mul_grid", {gridConst, zoom}, {grid});
    Node gs      = makeNode(OpType::GridSample, "gs", {scaled, grid}, {y});
    g.nodes.push_back(mulData);
    g.nodes.push_back(mulGrid);
    g.nodes.push_back(gs);

    const Fp16ConvertStats st = convertInitializersFp16(g);
    EXPECT_EQ(g.desc(weight).dtype, DType::Float16) << "a data-path constant must still narrow";
    EXPECT_EQ(g.desc(gridConst).dtype, DType::Float32) << "a grid-algebra constant keeps fp32";
    EXPECT_EQ(g.desc(zoom).dtype, DType::Float32) << "a scalar in the grid algebra keeps fp32";
    EXPECT_EQ(st.keptCoord, 2);
    EXPECT_EQ(st.converted, 1);
}

TEST(ConvertFp16Coord, SpatialAxisRampKeepsByShapeBiasNarrows) {
    Graph    g;
    TensorId x    = addAct(g, "x", {1, 4, 64, 96});
    TensorId ramp = addFloatInit(g, "ramp_w", {1, 1, 1, 96}); // trailing-axis ramp: kept
    TensorId bias = addFloatInit(g, "bias", {64});            // rank-1 magnitude: narrows
    TensorId y    = addAct(g, "y", {1, 4, 64, 96});
    TensorId z    = addAct(g, "z", {1, 4, 64, 96});
    g.nodes.push_back(makeNode(OpType::Binary, "mask", {x, ramp}, {y}));
    g.nodes.push_back(makeNode(OpType::Add, "shift", {y, bias}, {z}));

    const Fp16ConvertStats st = convertInitializersFp16(g);
    EXPECT_EQ(g.desc(ramp).dtype, DType::Float32) << "a spatial-axis ramp keeps fp32 by shape";
    EXPECT_EQ(g.desc(bias).dtype, DType::Float16) << "a rank-1 bias narrows";
    EXPECT_EQ(st.keptCoord, 1);
    EXPECT_EQ(st.converted, 1);
}
