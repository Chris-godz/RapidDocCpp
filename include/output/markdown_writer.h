#pragma once

#include "common/types.h"
#include <string>

namespace rapid_doc {

class MarkdownWriter {
public:
    std::string generate(const DocumentResult& result) const;

private:
    std::string elementToMarkdown(const ContentElement& elem) const;
};

} // namespace rapid_doc