#pragma once

// -----------------------------------------------------------------------
// corpus.h
//   Training data for the expert models.
//
//   Every expert domain has a corpus: `data/corpus/<domain>.txt` (one
//   sentence per line, optionally "<importance>\t<text>").  If the file is
//   missing, a synthetic corpus of ~target_words words is generated from
//   domain templates and written there so it can be edited / replaced
//   with real text.  Set OPENRAI_CORPUS_DIR to use your own directory.
// -----------------------------------------------------------------------

#include <string>
#include <vector>

namespace openrai {

struct CorpusLine {
    std::string text;
    float       importance = 0.f;   // 0..1 label used by the MLP scorer
};

// Synthetic corpus for a built-in domain (stability, anomaly, memory,
// throughput, planning, reflection, telemetry, language, math).  Unknown
// domains get a generic "system" corpus.
std::vector<CorpusLine> generate_domain_corpus(const std::string& domain, size_t target_words, unsigned seed);

// Same templates, but an exact number of lines (used for large fixed-size
// corpora where "500000 expressions" should mean exactly that).
std::vector<CorpusLine> generate_domain_corpus_lines(const std::string& domain, size_t n_lines, unsigned seed);

// Held-out arithmetic questions (prompt ends right before the answer, e.g.
// "what is 12 plus 7 equals") with the ground-truth integer result - for
// evaluating a math model by generation only, without a parser.
struct MathQuestion { std::string prompt; long long expected = 0; };
std::vector<MathQuestion> generate_math_questions(size_t n, unsigned seed);

// Load "<importance>\t<text>" or plain text lines.
std::vector<CorpusLine> load_corpus_file(const std::string& path);
bool save_corpus_file(const std::string& path, const std::vector<CorpusLine>& lines);

// Load or generate+save.
std::vector<CorpusLine> load_or_generate_corpus(const std::string& dir, const std::string& domain,
                                                size_t target_words, unsigned seed);

size_t corpus_word_count(const std::vector<CorpusLine>& lines);
std::vector<std::string> corpus_texts(const std::vector<CorpusLine>& lines);

} // namespace openrai
