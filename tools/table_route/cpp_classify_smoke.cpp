// Parity smoke for TableClassifier (PaddleCls port).
//
// Usage: cpp_classify_smoke <paddle_cls.onnx> <crop1.png> [<crop2.png> ...]
//
// Prints one JSON-like line per crop:
//   {"path": "...", "label": "wired", "score": 0.9876, "infer_ms": 4.2}
//
// Goal: confirm C++ TableClassifier agrees with Python PaddleCls argmax on
// the 3 table crops dumped from test_files/表格2.pdf.

#include "table/table_classifier.h"

#include <opencv2/opencv.hpp>

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: %s <paddle_cls.onnx> <crop1.png> [<crop2.png> ...]\n",
                     argv[0]);
        return 2;
    }

    rapid_doc::TableClassifierConfig cfg;
    cfg.onnxModelPath = argv[1];
    cfg.intraOpThreads = -1;
    cfg.disableCpuMemArena = true;

    rapid_doc::TableClassifier classifier(cfg);
    if (!classifier.initialize()) {
        std::fprintf(stderr, "failed to initialize TableClassifier\n");
        return 3;
    }

    for (int i = 2; i < argc; ++i) {
        const std::string path = argv[i];
        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR); // BGR, parity with pipeline
        if (img.empty()) {
            std::printf("{\"path\":\"%s\",\"error\":\"empty_image\"}\n", path.c_str());
            continue;
        }
        rapid_doc::TableClassifyResult r = classifier.classify(img);
        std::printf(
            "{\"path\":\"%s\",\"ok\":%s,\"label\":\"%s\",\"pred_index\":%d,"
            "\"score\":%.6f,\"logit_wired\":%.6f,\"logit_wireless\":%.6f,"
            "\"preprocess_ms\":%.3f,\"infer_ms\":%.3f}\n",
            path.c_str(),
            r.ok ? "true" : "false",
            r.label.c_str(),
            r.predIndex,
            r.score,
            r.rawLogitWired,
            r.rawLogitWireless,
            r.preprocessMs,
            r.inferMs);
        std::fflush(stdout);
    }
    return 0;
}
