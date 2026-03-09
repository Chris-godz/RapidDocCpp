/**
 * @file test_output.cpp
 * @brief Tests for output modules (MarkdownWriter, ContentListWriter)
 *
 * Unit tests for markdown generation and JSON content list output.
 */

#include <gtest/gtest.h>
#include "output/markdown_writer.h"
#include "output/content_list.h"
#include <opencv2/opencv.hpp>

namespace {

rapid_doc::ContentElement makeElement(rapid_doc::ContentElement::Type type, const std::string& text = "") {
    rapid_doc::ContentElement elem;
    elem.type = type;
    elem.text = text;
    elem.layoutBox = {10, 10, 100, 50, rapid_doc::LayoutCategory::TEXT, 0.9f, 0};
    elem.pageIndex = 0;
    elem.confidence = 0.95f;
    return elem;
}

rapid_doc::PageResult makePage(const std::vector<rapid_doc::ContentElement>& elements) {
    rapid_doc::PageResult page;
    page.pageIndex = 0;
    page.elements = elements;
    return page;
}

rapid_doc::DocumentResult makeDoc(const std::vector<rapid_doc::PageResult>& pages) {
    rapid_doc::DocumentResult doc;
    doc.pages = pages;
    doc.totalPages = static_cast<int>(pages.size());
    doc.processedPages = static_cast<int>(pages.size());
    return doc;
}

} // anonymous namespace

// ========================================
// MarkdownWriter Tests
// ========================================

TEST(MarkdownWriter, DefaultOptions) {
    rapid_doc::MarkdownOptions opts;
    EXPECT_TRUE(opts.includeImages);
    EXPECT_TRUE(opts.includeTables);
    EXPECT_FALSE(opts.includeUnsupported);
    EXPECT_EQ(opts.maxTitleLevel, 4);
    EXPECT_TRUE(opts.escapeSpecialChars);
}

TEST(MarkdownWriter, GenerateTextElement) {
    rapid_doc::MarkdownWriter writer;
    rapid_doc::ContentElement elem = makeElement(rapid_doc::ContentElement::Type::TEXT, "Hello World");
    
    rapid_doc::PageResult page = makePage({elem});
    std::string md = writer.generatePage(page);
    
    EXPECT_EQ(md, "Hello World\n\n");
}

TEST(MarkdownWriter, GenerateTitleElement) {
    rapid_doc::MarkdownWriter writer;
    rapid_doc::ContentElement elem = makeElement(rapid_doc::ContentElement::Type::TITLE, "Document Title");
    elem.readingOrder = 1;
    
    rapid_doc::PageResult page = makePage({elem});
    std::string md = writer.generatePage(page);
    
    EXPECT_EQ(md, "# Document Title\n\n");
}

TEST(MarkdownWriter, TitleLevelClamped) {
    rapid_doc::MarkdownWriter writer;
    rapid_doc::ContentElement elem = makeElement(rapid_doc::ContentElement::Type::TITLE, "Low Level");
    elem.readingOrder = 10;
    
    rapid_doc::MarkdownOptions opts;
    opts.maxTitleLevel = 4;
    rapid_doc::MarkdownWriter writer2(opts);
    
    rapid_doc::PageResult page = makePage({elem});
    std::string md = writer2.generatePage(page);
    
    EXPECT_EQ(md, "#### Low Level\n\n");
}

TEST(MarkdownWriter, GenerateImageElement) {
    rapid_doc::MarkdownOptions opts;
    opts.includeImages = true;
    rapid_doc::MarkdownWriter writer(opts);
    
    rapid_doc::ContentElement elem = makeElement(rapid_doc::ContentElement::Type::IMAGE);
    elem.imagePath = "page0_fig0.png";
    
    rapid_doc::PageResult page = makePage({elem});
    std::string md = writer.generatePage(page);
    
    EXPECT_EQ(md, "![](page0_fig0.png)\n\n");
}

TEST(MarkdownWriter, GenerateTableElement) {
    rapid_doc::MarkdownOptions opts;
    opts.includeTables = true;
    rapid_doc::MarkdownWriter writer(opts);
    
    rapid_doc::ContentElement elem = makeElement(rapid_doc::ContentElement::Type::TABLE);
    elem.html = "<table><tr><td>Cell</td></tr></table>";
    
    rapid_doc::PageResult page = makePage({elem});
    std::string md = writer.generatePage(page);
    
    EXPECT_EQ(md, "<table><tr><td>Cell</td></tr></table>\n\n");
}

TEST(MarkdownWriter, GenerateCodeElement) {
    rapid_doc::MarkdownWriter writer;
    rapid_doc::ContentElement elem = makeElement(rapid_doc::ContentElement::Type::CODE, "int main() {}");
    
    rapid_doc::PageResult page = makePage({elem});
    std::string md = writer.generatePage(page);
    
    EXPECT_EQ(md, "```\nint main() {}\n```\n\n");
}

TEST(MarkdownWriter, SkippedElementsExcluded) {
    rapid_doc::MarkdownOptions opts;
    opts.includeUnsupported = false;
    rapid_doc::MarkdownWriter writer(opts);
    
    rapid_doc::ContentElement elem = makeElement(rapid_doc::ContentElement::Type::TEXT, "Visible");
    elem.skipped = true;
    
    rapid_doc::PageResult page = makePage({elem});
    std::string md = writer.generatePage(page);
    
    EXPECT_TRUE(md.empty());
}

TEST(MarkdownWriter, SkippedElementsIncludedAsComment) {
    rapid_doc::MarkdownOptions opts;
    opts.includeUnsupported = true;
    rapid_doc::MarkdownWriter writer(opts);
    
    rapid_doc::ContentElement elem = makeElement(rapid_doc::ContentElement::Type::TEXT, "Visible");
    elem.skipped = true;
    elem.layoutBox.category = rapid_doc::LayoutCategory::FORMULA;
    
    rapid_doc::PageResult page = makePage({elem});
    std::string md = writer.generatePage(page);
    
    EXPECT_FALSE(md.empty());
    EXPECT_NE(md.find("UNSUPPORTED"), std::string::npos);
}

TEST(MarkdownWriter, EscapeSpecialChars) {
    rapid_doc::MarkdownOptions opts;
    opts.escapeSpecialChars = true;
    rapid_doc::MarkdownWriter writer(opts);
    
    rapid_doc::ContentElement elem = makeElement(rapid_doc::ContentElement::Type::TEXT, "Test *bold* and `code`");
    
    rapid_doc::PageResult page = makePage({elem});
    std::string md = writer.generatePage(page);
    
    EXPECT_NE(md.find("\\*"), std::string::npos);
    EXPECT_NE(md.find("\\`"), std::string::npos);
}

TEST(MarkdownWriter, EscapeTildeAndDollar) {
    rapid_doc::MarkdownOptions opts;
    opts.escapeSpecialChars = true;
    rapid_doc::MarkdownWriter writer(opts);
    
    rapid_doc::ContentElement elem = makeElement(rapid_doc::ContentElement::Type::TEXT, "Test ~tilde~ and $dollar$");
    
    rapid_doc::PageResult page = makePage({elem});
    std::string md = writer.generatePage(page);
    
    EXPECT_NE(md.find("\\~"), std::string::npos);
    EXPECT_NE(md.find("\\$"), std::string::npos);
}

TEST(MarkdownWriter, EmptyPage) {
    rapid_doc::MarkdownWriter writer;
    rapid_doc::PageResult page = makePage({});
    std::string md = writer.generatePage(page);
    EXPECT_TRUE(md.empty());
}

TEST(MarkdownWriter, MultiPageDocument) {
    rapid_doc::MarkdownWriter writer;
    
    rapid_doc::PageResult page1 = makePage({makeElement(rapid_doc::ContentElement::Type::TEXT, "Page 1")});
    rapid_doc::PageResult page2 = makePage({makeElement(rapid_doc::ContentElement::Type::TEXT, "Page 2")});
    
    rapid_doc::DocumentResult doc = makeDoc({page1, page2});
    std::string md = writer.generate(doc);
    
    EXPECT_NE(md.find("Page 1"), std::string::npos);
    EXPECT_NE(md.find("Page 2"), std::string::npos);
    EXPECT_NE(md.find("---"), std::string::npos);
}

// ========================================
// ContentListWriter Tests
// ========================================

TEST(ContentListWriter, GenerateEmptyPage) {
    rapid_doc::ContentListWriter writer;
    rapid_doc::PageResult page;
    page.pageIndex = 0;
    
    std::string json = writer.generatePage(page, 1000, 1000);
    
    EXPECT_FALSE(json.empty());  // Should be valid JSON array
    EXPECT_NE(json.find("["), std::string::npos);
}

TEST(ContentListWriter, GenerateTextElement) {
    rapid_doc::ContentListWriter writer;
    rapid_doc::ContentElement elem = makeElement(rapid_doc::ContentElement::Type::TEXT, "Hello");
    
    rapid_doc::PageResult page = makePage({elem});
    std::string json = writer.generatePage(page, 1000, 1000);
    
    EXPECT_NE(json.find("text"), std::string::npos);
    EXPECT_NE(json.find("Hello"), std::string::npos);
    EXPECT_NE(json.find("\"type\": \"text\""), std::string::npos);
}

TEST(ContentListWriter, GenerateTitleElement) {
    rapid_doc::ContentListWriter writer;
    rapid_doc::ContentElement elem = makeElement(rapid_doc::ContentElement::Type::TITLE, "Title");
    elem.readingOrder = 2;
    
    rapid_doc::PageResult page = makePage({elem});
    std::string json = writer.generatePage(page, 1000, 1000);
    
    EXPECT_NE(json.find("\"type\": \"title\""), std::string::npos);
    EXPECT_NE(json.find("text_level"), std::string::npos);
}

TEST(ContentListWriter, GenerateImageElement) {
    rapid_doc::ContentListWriter writer;
    rapid_doc::ContentElement elem = makeElement(rapid_doc::ContentElement::Type::IMAGE);
    elem.imagePath = "fig1.png";
    
    rapid_doc::PageResult page = makePage({elem});
    std::string json = writer.generatePage(page, 1000, 1000);
    
    EXPECT_NE(json.find("img_path"), std::string::npos);
    EXPECT_NE(json.find("fig1.png"), std::string::npos);
}

TEST(ContentListWriter, GenerateTableElement) {
    rapid_doc::ContentListWriter writer;
    rapid_doc::ContentElement elem = makeElement(rapid_doc::ContentElement::Type::TABLE);
    elem.html = "<table>...</table>";
    
    rapid_doc::PageResult page = makePage({elem});
    std::string json = writer.generatePage(page, 1000, 1000);
    
    EXPECT_NE(json.find("html"), std::string::npos);
    EXPECT_NE(json.find("<table>...</table>"), std::string::npos);
}

TEST(ContentListWriter, BboxNormalized) {
    rapid_doc::ContentListWriter writer;
    rapid_doc::ContentElement elem = makeElement(rapid_doc::ContentElement::Type::TEXT, "Test");
    elem.layoutBox = {100, 100, 200, 200, rapid_doc::LayoutCategory::TEXT, 0.9f, 0};
    
    rapid_doc::PageResult page = makePage({elem});
    std::string json = writer.generatePage(page, 1000, 1000);
    
    EXPECT_NE(json.find("bbox"), std::string::npos);
}

TEST(ContentListWriter, SkippedElementFlag) {
    rapid_doc::ContentListWriter writer;
    rapid_doc::ContentElement elem = makeElement(rapid_doc::ContentElement::Type::TEXT, "Skipped");
    elem.skipped = true;
    
    rapid_doc::PageResult page = makePage({elem});
    std::string json = writer.generatePage(page, 1000, 1000);
    
    EXPECT_NE(json.find("\"skipped\": true"), std::string::npos);
    EXPECT_NE(json.find("NPU unsupported"), std::string::npos);
}

TEST(ContentListWriter, PageIndexIncluded) {
    rapid_doc::ContentListWriter writer;
    rapid_doc::ContentElement elem = makeElement(rapid_doc::ContentElement::Type::TEXT, "Test");
    elem.pageIndex = 5;
    
    rapid_doc::PageResult page = makePage({elem});
    std::string json = writer.generatePage(page, 1000, 1000);
    
    EXPECT_NE(json.find("\"page_idx\": 5"), std::string::npos);
}

TEST(ContentListWriter, AllElementTypes) {
    rapid_doc::ContentListWriter writer;
    
    std::vector<rapid_doc::ContentElement> elems;
    elems.push_back(makeElement(rapid_doc::ContentElement::Type::TEXT, "text"));
    elems.push_back(makeElement(rapid_doc::ContentElement::Type::TITLE, "title"));
    elems.push_back(makeElement(rapid_doc::ContentElement::Type::IMAGE, ""));
    elems.push_back(makeElement(rapid_doc::ContentElement::Type::TABLE, ""));
    elems.push_back(makeElement(rapid_doc::ContentElement::Type::EQUATION, ""));
    elems.push_back(makeElement(rapid_doc::ContentElement::Type::CODE, ""));
    elems.push_back(makeElement(rapid_doc::ContentElement::Type::LIST, ""));
    elems.push_back(makeElement(rapid_doc::ContentElement::Type::UNKNOWN, ""));
    
    rapid_doc::PageResult page = makePage(elems);
    std::string json = writer.generatePage(page, 1000, 1000);
    
    EXPECT_NE(json.find("\"type\": \"text\""), std::string::npos);
    EXPECT_NE(json.find("\"type\": \"title\""), std::string::npos);
    EXPECT_NE(json.find("\"type\": \"image\""), std::string::npos);
    EXPECT_NE(json.find("\"type\": \"table\""), std::string::npos);
    EXPECT_NE(json.find("\"type\": \"equation\""), std::string::npos);
    EXPECT_NE(json.find("\"type\": \"code\""), std::string::npos);
    EXPECT_NE(json.find("\"type\": \"list\""), std::string::npos);
    EXPECT_NE(json.find("\"type\": \"unknown\""), std::string::npos);
}
