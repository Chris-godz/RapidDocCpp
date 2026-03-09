#include "common/types.h"
#include <algorithm>

namespace rapid_doc {

const char* layoutCategoryToString(LayoutCategory cat) {
    switch (cat) {
        // PP-DocLayout-L DXEngine 23 categories
        case LayoutCategory::PARAGRAPH_TITLE:    return "paragraph_title";
        case LayoutCategory::IMAGE:              return "image";
        case LayoutCategory::TEXT:               return "text";
        case LayoutCategory::NUMBER:             return "number";
        case LayoutCategory::ABSTRACT:           return "abstract";
        case LayoutCategory::CONTENT:            return "content";
        case LayoutCategory::FIGURE_TITLE:       return "figure_title";
        case LayoutCategory::FORMULA:            return "formula";
        case LayoutCategory::TABLE:              return "table";
        case LayoutCategory::TABLE_TITLE:        return "table_title";
        case LayoutCategory::REFERENCE:          return "reference";
        case LayoutCategory::DOC_TITLE:          return "doc_title";
        case LayoutCategory::FOOTNOTE:           return "footnote";
        case LayoutCategory::HEADER:             return "header";
        case LayoutCategory::ALGORITHM:          return "algorithm";
        case LayoutCategory::FOOTER:             return "footer";
        case LayoutCategory::SEAL:               return "seal";
        case LayoutCategory::CHART_TITLE:        return "chart_title";
        case LayoutCategory::CHART:              return "chart";
        case LayoutCategory::FORMULA_NUMBER:     return "formula_number";
        case LayoutCategory::HEADER_IMAGE:       return "header_image";
        case LayoutCategory::FOOTER_IMAGE:       return "footer_image";
        case LayoutCategory::ASIDE_TEXT:         return "aside_text";
        default:                                 return "unknown";
    }
}

bool isCategorySupported(LayoutCategory cat) {
    switch (cat) {
        case LayoutCategory::FORMULA:
        case LayoutCategory::FORMULA_NUMBER:
            // Formula recognition not supported on DEEPX NPU
            return false;
        case LayoutCategory::HEADER_IMAGE:
        case LayoutCategory::FOOTER_IMAGE:
        case LayoutCategory::SEAL:
        case LayoutCategory::NUMBER:
        case LayoutCategory::FOOTNOTE:
        case LayoutCategory::HEADER:
        case LayoutCategory::FOOTER:
            // Categorized as "Abandon" in Python RapidDoc — skip processing
            return false;
        default:
            return true;
    }
}

std::vector<LayoutBox> LayoutResult::getBoxesByCategory(LayoutCategory cat) const {
    std::vector<LayoutBox> result;
    std::copy_if(boxes.begin(), boxes.end(), std::back_inserter(result),
        [cat](const LayoutBox& b) { return b.category == cat; });
    return result;
}

std::vector<LayoutBox> LayoutResult::getTextBoxes() const {
    std::vector<LayoutBox> result;
    std::copy_if(boxes.begin(), boxes.end(), std::back_inserter(result),
        [](const LayoutBox& b) {
            // Text-like categories that need OCR
            return b.category == LayoutCategory::TEXT ||
                   b.category == LayoutCategory::PARAGRAPH_TITLE ||
                   b.category == LayoutCategory::DOC_TITLE ||
                   b.category == LayoutCategory::ABSTRACT ||
                   b.category == LayoutCategory::CONTENT ||
                   b.category == LayoutCategory::REFERENCE ||
                   b.category == LayoutCategory::ALGORITHM ||
                   b.category == LayoutCategory::ASIDE_TEXT ||
                   // Caption categories — contain text that should be OCR'd
                   b.category == LayoutCategory::FIGURE_TITLE ||
                   b.category == LayoutCategory::TABLE_TITLE ||
                   b.category == LayoutCategory::CHART_TITLE;
        });
    return result;
}

std::vector<LayoutBox> LayoutResult::getTableBoxes() const {
    return getBoxesByCategory(LayoutCategory::TABLE);
}

std::vector<LayoutBox> LayoutResult::getSupportedBoxes() const {
    std::vector<LayoutBox> result;
    std::copy_if(boxes.begin(), boxes.end(), std::back_inserter(result),
        [](const LayoutBox& b) { return isCategorySupported(b.category); });
    return result;
}

std::vector<LayoutBox> LayoutResult::getUnsupportedBoxes() const {
    std::vector<LayoutBox> result;
    std::copy_if(boxes.begin(), boxes.end(), std::back_inserter(result),
        [](const LayoutBox& b) { return !isCategorySupported(b.category); });
    return result;
}

ContentElement::NormalizedBBox ContentElement::getNormalizedBBox(int pageWidth, int pageHeight) const {
    NormalizedBBox nb;
    nb.x0 = static_cast<int>(layoutBox.x0 / pageWidth * 1000);
    nb.y0 = static_cast<int>(layoutBox.y0 / pageHeight * 1000);
    nb.x1 = static_cast<int>(layoutBox.x1 / pageWidth * 1000);
    nb.y1 = static_cast<int>(layoutBox.y1 / pageHeight * 1000);
    return nb;
}

} // namespace rapid_doc
