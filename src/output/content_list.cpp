#include "output/content_list.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace rapid_doc {

static const char* typeToString(ContentElement::Type t) {
    switch (t) {
    case ContentElement::Type::TEXT:      return "text";
    case ContentElement::Type::TITLE:     return "title";
    case ContentElement::Type::IMAGE:     return "image";
    case ContentElement::Type::TABLE:     return "table";
    case ContentElement::Type::EQUATION:  return "equation";
    case ContentElement::Type::CODE:      return "code";
    case ContentElement::Type::LIST:      return "list";
    case ContentElement::Type::HEADER:    return "header";
    case ContentElement::Type::FOOTER:    return "footer";
    case ContentElement::Type::REFERENCE: return "reference";
    default:                              return "unknown";
    }
}

std::string ContentListWriter::generate(const DocumentResult& result) const {
    json doc = json::array();

    for (const auto& page : result.pages) {
        json pageArr = json::array();
        for (const auto& elem : page.elements) {
            json item;
            item["type"]    = typeToString(elem.type);
            item["text"]    = elem.text;
            item["page"]    = elem.pageIndex;
            item["order"]   = elem.readingOrder;
            item["skipped"] = elem.skipped;
            item["bbox"]    = {elem.layoutBox.x0, elem.layoutBox.y0,
                               elem.layoutBox.x1, elem.layoutBox.y1};
            if (!elem.html.empty())
                item["html"] = elem.html;
            if (!elem.imagePath.empty())
                item["image_path"] = elem.imagePath;
            pageArr.push_back(std::move(item));
        }
        doc.push_back(std::move(pageArr));
    }

    return doc.dump(2);
}

} // namespace rapid_doc
