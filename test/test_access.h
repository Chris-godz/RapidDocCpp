#pragma once

#include "pipeline/doc_pipeline.h"
#include "server/server.h"
#include <mutex>

namespace rapid_doc {

class DocPipelineTestAccess {
public:
    static void setOcrHooks(
        DocPipeline& pipeline,
        std::function<bool(const cv::Mat&, int64_t)> submitHook,
        std::function<bool(std::vector<ocr::PipelineOCRResult>&, int64_t&, bool&)> fetchHook)
    {
        pipeline.ocrSubmitHook_ = std::move(submitHook);
        pipeline.ocrFetchHook_ = std::move(fetchHook);
    }

    static void clearOcrHooks(DocPipeline& pipeline) {
        pipeline.ocrSubmitHook_ = {};
        pipeline.ocrFetchHook_ = {};
        std::lock_guard<std::mutex> lock(pipeline.ocrStateMutex_);
        pipeline.bufferedOcrResults_.clear();
        pipeline.timedOutOcrTaskIds_.clear();
    }

    static void setTableHooks(
        DocPipeline& pipeline,
        std::function<TableResult(const cv::Mat&)> recognizeHook,
        std::function<std::string(const std::vector<TableCell>&)> htmlHook = {})
    {
        pipeline.tableRecognizeHook_ = std::move(recognizeHook);
        pipeline.tableHtmlHook_ = std::move(htmlHook);
    }

    static void clearTableHooks(DocPipeline& pipeline) {
        pipeline.tableRecognizeHook_ = {};
        pipeline.tableHtmlHook_ = {};
    }

    static void setOcrTimeout(DocPipeline& pipeline, std::chrono::milliseconds timeout) {
        if (timeout.count() <= 0) {
            pipeline.ocrWaitTimeout_ = std::chrono::milliseconds(1);
            return;
        }
        pipeline.ocrWaitTimeout_ = timeout;
    }

    static std::string ocrOnCrop(DocPipeline& pipeline, const cv::Mat& crop, int64_t taskId) {
        return pipeline.ocrOnCrop(crop, taskId);
    }

    static std::vector<ContentElement> runOcrOnRegions(
        DocPipeline& pipeline,
        const cv::Mat& image,
        const std::vector<LayoutBox>& textBoxes,
        int pageIndex)
    {
        return pipeline.runOcrOnRegions(image, textBoxes, pageIndex);
    }

    static std::vector<ContentElement> runTableRecognition(
        DocPipeline& pipeline,
        const cv::Mat& image,
        const std::vector<LayoutBox>& tableBoxes,
        int pageIndex)
    {
        return pipeline.runTableRecognition(image, tableBoxes, pageIndex);
    }

    static PageResult processPageWithLayoutBoxes(
        DocPipeline& pipeline,
        const cv::Mat& image,
        const std::vector<LayoutBox>& layoutBoxes,
        int pageIndex)
    {
        pipeline.resetOcrTransientStateForRun();
        auto ctx = pipeline.makeExecutionContext(nullptr);

        DocPipeline::PageContext pageCtx;
        pageCtx.wallStartTime = std::chrono::steady_clock::now();
        pageCtx.pageImage.image = image;
        pageCtx.pageImage.pageIndex = pageIndex;
        pageCtx.pageImage.dpi = pipeline.config_.runtime.pdfDpi;
        pageCtx.pageImage.scaleFactor = 1.0;
        pageCtx.pageImage.pdfWidth = image.cols;
        pageCtx.pageImage.pdfHeight = image.rows;
        pageCtx.pageResult.pageIndex = pageIndex;
        pageCtx.pageResult.pageWidth = image.cols;
        pageCtx.pageResult.pageHeight = image.rows;
        pageCtx.pageResult.layoutResult.boxes = layoutBoxes;

        pipeline.runPlanStage(pageCtx, ctx);
        pipeline.runOcrTableStage(pageCtx, ctx);
        pipeline.runFinalizeStage(pageCtx, ctx);
        return pageCtx.pageResult;
    }

    static std::vector<ContentElement> handleUnsupportedElements(
        DocPipeline& pipeline,
        const std::vector<LayoutBox>& unsupportedBoxes,
        int pageIndex)
    {
        return pipeline.handleUnsupportedElements(unsupportedBoxes, pageIndex);
    }

    static void saveExtractedImages(
        DocPipeline& pipeline,
        const cv::Mat& image,
        const std::vector<LayoutBox>& figureBoxes,
        int pageIndex,
        std::vector<ContentElement>& elements)
    {
        pipeline.saveExtractedImages(image, figureBoxes, pageIndex, elements);
    }

    static void saveFormulaImages(
        DocPipeline& pipeline,
        const cv::Mat& image,
        const std::vector<LayoutBox>& equationBoxes,
        int pageIndex,
        std::vector<ContentElement>& elements)
    {
        pipeline.saveFormulaImages(image, equationBoxes, pageIndex, elements);
    }
};

class DocServerTestAccess {
public:
    static std::string handleProcess(
        DocServer& server,
        const std::string& pdfData,
        const std::string& filename)
    {
        return server.handleProcess(pdfData, filename);
    }

    static DocPipeline& pipeline(DocServer& server) {
        return *server.shards_.front()->pipeline;
    }
};

} // namespace rapid_doc
