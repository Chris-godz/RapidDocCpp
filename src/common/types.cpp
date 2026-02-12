#include "common/types.h"
#include <algorithm>

namespace rapid_doc {

const char* layoutCategoryToString(LayoutCategory cat) {
    switch (cat) {
        case LayoutCategory::TEXT:               return "text";
        case LayoutCategory::TITLE:              return "title";
        case LayoutCategory::FIGURE:             return "figure";
        case LayoutCategory::FIGURE_CAPTION:     return "figure_caption";
        case LayoutCategory::TABLE:              return "table";
        case LayoutCategory::TABLE_CAPTION:      return "table_caption";
        case LayoutCategory::TABLE_FOOTNOTE:     return "table_footnote";
        case LayoutCategory::HEADER:             return "header";
        case LayoutCategory::FOOTER:             return "footer";
        case LayoutCategory::REFERENCE:          return "reference";
        case LayoutCategory::EQUATION:           return "equation";
        case LayoutCategory::INTERLINE_EQUATION: return "interline_equation";
        case LayoutCategory::STAMP:              return "stamp";
        case LayoutCategory::CODE:               return "code";
        case LayoutCategory::TOC:                return "toc";
        case LayoutCategory::ABSTRACT:           return "abstract";
        case LayoutCategory::CONTENT:            return "content";
        case LayoutCategory::LIST:               return "list";
        case LayoutCategory::INDEX:              return "index";
        case LayoutCategory::SEPARATOR:          return "separator";
        default:                                 return "unknown";
    }
}

bool isCategorySupported(LayoutCategory cat) {
    switch (cat) {
        case LayoutCategory::EQUATION:
        case LayoutCategory::INTERLINE_EQUATION:
            // Formula recognition not supported on DEEPX NPU
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
            return b.category == LayoutCategory::TEXT ||
                   b.category == LayoutCategory::TITLE ||
                   b.category == LayoutCategory::CONTENT ||
                   b.category == LayoutCategory::LIST ||
                   b.category == LayoutCategory::CODE ||
                   b.category == LayoutCategory::ABSTRACT ||
                   b.category == LayoutCategory::REFERENCE ||
                   b.category == LayoutCategory::INDEX ||
                   b.category == LayoutCategory::HEADER ||
                   b.category == LayoutCategory::FOOTER;
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
