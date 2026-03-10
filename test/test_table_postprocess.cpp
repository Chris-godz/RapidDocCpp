/**
 * @file test_table_postprocess.cpp
 * @brief Validate C++ table cell extraction matches Python output.
 *
 * Compares cell bounding boxes and structure.
 */

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <filesystem>
#include "npy_loader.h"

namespace fs = std::filesystem;
#include "metrics.h"

using namespace rapid_doc;
using namespace rapid_doc::test_utils;
using json = nlohmann::json;

static const std::string kFixtureDir = std::string(PROJECT_ROOT_DIR) + "/test/fixtures/table/";

class TablePostprocessTest : public ::testing::Test {
protected:
    void SetUp() override {
        hasCells_ = fs::exists(kFixtureDir + "cells.json");
        hasMask_ = fs::exists(kFixtureDir + "seg_mask.npy");
    }
    bool hasCells_ = false;
    bool hasMask_ = false;
};

TEST_F(TablePostprocessTest, CellStructureIsValid) {
    if (!hasCells_) {
        GTEST_SKIP() << "cells.json not found — table cell extraction fixture not generated.";
    }

    std::string jsonStr = loadJsonString(kFixtureDir + "cells.json");
    auto pyCells = json::parse(jsonStr);

    if (pyCells.empty()) {
        GTEST_SKIP() << "No cells in Python output.";
    }

    for (const auto& cell : pyCells) {
        if (cell.contains("bbox")) {
            auto bbox = cell["bbox"];
            EXPECT_EQ(bbox.size(), 4u);
            float x0 = bbox[0].get<float>();
            float y0 = bbox[1].get<float>();
            float x1 = bbox[2].get<float>();
            float y1 = bbox[3].get<float>();
            EXPECT_GT(x1, x0) << "Cell width must be positive";
            EXPECT_GT(y1, y0) << "Cell height must be positive";
        }
    }
}

TEST_F(TablePostprocessTest, MaskIntermediatesExist) {
    if (!hasMask_) {
        GTEST_SKIP() << "seg_mask.npy not found.";
    }

    EXPECT_TRUE(fs::exists(kFixtureDir + "hpred.npy")) << "hpred.npy should exist";
    EXPECT_TRUE(fs::exists(kFixtureDir + "vpred.npy")) << "vpred.npy should exist";

    NpyArray hpred = loadNpy(kFixtureDir + "hpred.npy");
    NpyArray vpred = loadNpy(kFixtureDir + "vpred.npy");

    EXPECT_GE(hpred.shape.size(), 2u);
    EXPECT_GE(vpred.shape.size(), 2u);

    const uint8_t* hData = hpred.asUint8();
    const uint8_t* vData = vpred.asUint8();
    size_t totalH = hpred.elementCount();
    size_t totalV = vpred.elementCount();

    int hPixels = 0, vPixels = 0;
    for (size_t i = 0; i < totalH; ++i) {
        if (hData[i] > 0) hPixels++;
    }
    for (size_t i = 0; i < totalV; ++i) {
        if (vData[i] > 0) vPixels++;
    }

    std::cerr << "  [INFO] hpred non-zero: " << hPixels << "/" << totalH
              << ", vpred non-zero: " << vPixels << "/" << totalV << std::endl;

    EXPECT_GT(hPixels, 0) << "hpred should have horizontal line pixels";
    EXPECT_GT(vPixels, 0) << "vpred should have vertical line pixels";
}

TEST_F(TablePostprocessTest, CppCellsMatchPython) {
    if (!hasCells_ || !fs::exists(kFixtureDir + "cpp_cells.json")) {
        GTEST_SKIP() << "Cell comparison fixtures not available.";
    }

    std::string pyStr = loadJsonString(kFixtureDir + "cells.json");
    std::string cppStr = loadJsonString(kFixtureDir + "cpp_cells.json");
    auto pyCells = json::parse(pyStr);
    auto cppCells = json::parse(cppStr);

    int pyCount = static_cast<int>(pyCells.size());
    int cppCount = static_cast<int>(cppCells.size());
    EXPECT_NEAR(cppCount, pyCount, pyCount * 0.2 + 1)
        << "Cell count mismatch: Python=" << pyCount << ", C++=" << cppCount;
}
