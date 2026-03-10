#include "output/markdown_writer.h"
#include <sstream>

namespace rapid_doc {

std::string MarkdownWriter::elementToMarkdown(const ContentElement& elem) const {
    if (elem.skipped && elem.text.empty() && elem.html.empty())
        return {};

    switch (elem.type) {
    case ContentElement::Type::TITLE: {
        // doc_title -> #, paragraph_title -> ##, others -> ##
        std::string prefix = "## ";
        if (elem.layoutBox.label == "doc_title")
            prefix = "# ";
        return prefix + elem.text + "\n\n";
    }

    case ContentElement::Type::TEXT:
    case ContentElement::Type::REFERENCE:
    case ContentElement::Type::LIST:
    case ContentElement::Type::CODE:
        if (elem.text.empty()) return {};
        return elem.text + "\n\n";

    case ContentElement::Type::TABLE:
        if (!elem.html.empty())
            return elem.html + "\n\n";
        return {};

    case ContentElement::Type::IMAGE:
        if (!elem.imagePath.empty())
            return "![](" + elem.imagePath + ")\n\n";
        return "![]()\n\n";

    case ContentElement::Type::EQUATION:
        // Python: formula rendered as image (no LaTeX on NPU)
        if (!elem.imagePath.empty())
            return "![](" + elem.imagePath + ")\n\n";
        if (!elem.text.empty())
            return elem.text + "\n\n";
        return {};

    case ContentElement::Type::HEADER:
    case ContentElement::Type::FOOTER:
        if (elem.text.empty()) return {};
        return elem.text + "\n\n";

    default:
        if (!elem.text.empty())
            return elem.text + "\n\n";
        return {};
    }
}

std::string MarkdownWriter::generate(const DocumentResult& result) const {
    std::ostringstream out;

    for (size_t pi = 0; pi < result.pages.size(); ++pi) {
        const auto& page = result.pages[pi];

        if (result.pages.size() > 1 && pi > 0)
            out << "\n---\n\n";

        for (const auto& elem : page.elements) {
            std::string md = elementToMarkdown(elem);
            if (!md.empty())
                out << md;
        }
    }

    return out.str();
}

} // namespace rapid_doc
