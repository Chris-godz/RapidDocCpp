#pragma once

#include "common/types.h"
#include <string>

namespace rapid_doc {

class ContentListWriter {
public:
    std::string generate(const DocumentResult& result) const;
};

} // namespace rapid_doc