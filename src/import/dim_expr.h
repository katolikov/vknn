// Evaluates an ONNX symbolic dimension expression (a Dimension.dim_param string) against a set of bound
// symbol values, so a dynamic input axis can be resolved from a few named-dim bindings instead of a full
// per-tensor shape. The grammar is small but covers what shape exporters emit for with-past decoders — a
// bare symbol ("past_sequence_length"), an integer literal, and sums/products of them
// ("past_sequence_length + sequence_length"):
//
//   expr   := term (('+' | '-') term)*
//   term   := factor ('*' factor)*
//   factor := integer | symbol | '(' expr ')'
//
// A symbol is any identifier ([A-Za-z_][A-Za-z0-9_.]*); '.' is allowed so a dotted name parses as one
// token. Whitespace is ignored. Every symbol not present in @p bindings is collected into `freeSymbols`
// (distinct, first-seen order) and marks the result not-ok; a malformed expression is likewise not-ok
// with the raw text recorded so the caller can name it in a diagnostic.
#pragma once
#include <cctype>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace vknn {

    /// True when a dim_param symbol NAMES the batch axis: "N", "B", or any name containing "batch"
    /// (case-insensitive) — the universal ONNX-export conventions. Only such a symbol may take the
    /// leading-axis `batch` fallback when unbound; any other leading symbol (num_frames, views,
    /// num_points...) is a real data extent whose silent freeze to batch=1 would truncate the caller's
    /// input, so shape resolution hard-errors on it and --list-dims reports it as must-bind.
    inline bool batchLikeDimSymbol(const std::string &sym) {
        std::string low;
        low.reserve(sym.size());
        for (char ch: sym)
        {
            low += (char) std::tolower((unsigned char) ch);
        }
        return low == "n" || low == "b" || low.find("batch") != std::string::npos;
    }

    /// Outcome of evaluating a symbolic dim expression: a resolved extent, or the set of unbound symbols
    /// that prevented resolution.
    struct DimEval {
        bool                     ok    = false; ///< true only when the whole expression resolved to a value
        int64_t                  value = 0;     ///< the resolved extent (meaningful only when ok)
        std::vector<std::string> freeSymbols;   ///< unbound symbols referenced (distinct, first-seen order)
    };

    namespace detail {
        // Recursive-descent evaluator over the dim-expression grammar. An unbound symbol contributes 0 to
        // the running value (so parsing continues and every free symbol is collected) and sets the result
        // not-ok via `freeSymbols`; a grammar error sets `bad`.
        struct DimExprParser {
            const std::string                    &s;
            const std::map<std::string, int64_t> &binds;
            std::vector<std::string>             &free;
            size_t                                i   = 0;
            bool                                  bad = false;

            DimExprParser(const std::string &str, const std::map<std::string, int64_t> &b, std::vector<std::string> &f)
                : s(str), binds(b), free(f) {
            }

            void        skipws() {
                while (i < s.size() && (s[i] == ' ' || s[i] == '\t'))
                {
                    ++i;
                }
            }
            static bool identStart(char c) {
                return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_';
            }
            static bool identChar(char c) {
                return identStart(c) || (c >= '0' && c <= '9') || c == '.';
            }
            void addFree(const std::string &name) {
                for (const auto &f0: free)
                {
                    if (f0 == name)
                    {
                        return;
                    }
                }
                free.push_back(name);
            }

            int64_t factor() {
                skipws();
                if (i >= s.size())
                {
                    bad = true;
                    return 0;
                }
                char c = s[i];
                if (c == '(')
                {
                    ++i;
                    int64_t v = expr();
                    skipws();
                    if (i < s.size() && s[i] == ')')
                    {
                        ++i;
                    } else
                    {
                        bad = true;
                    }
                    return v;
                }
                if (c >= '0' && c <= '9')
                {
                    int64_t v = 0;
                    while (i < s.size() && s[i] >= '0' && s[i] <= '9')
                    {
                        v = v * 10 + (s[i] - '0');
                        ++i;
                    }
                    return v;
                }
                if (identStart(c))
                {
                    size_t st = i;
                    while (i < s.size() && identChar(s[i]))
                    {
                        ++i;
                    }
                    std::string name = s.substr(st, i - st);
                    auto        it   = binds.find(name);
                    if (it == binds.end())
                    {
                        addFree(name);
                        return 0;
                    }
                    return it->second;
                }
                bad = true;
                return 0;
            }

            int64_t term() {
                int64_t v = factor();
                for (;;)
                {
                    skipws();
                    if (i < s.size() && s[i] == '*')
                    {
                        ++i;
                        v *= factor();
                    } else
                    {
                        break;
                    }
                }
                return v;
            }

            int64_t expr() {
                int64_t v = term();
                for (;;)
                {
                    skipws();
                    if (i < s.size() && s[i] == '+')
                    {
                        ++i;
                        v += term();
                    } else if (i < s.size() && s[i] == '-')
                    {
                        ++i;
                        v -= term();
                    } else
                    {
                        break;
                    }
                }
                return v;
            }
        };
    } // namespace detail

    /// Evaluate a symbolic dim expression @p expr against @p bindings. Returns ok only when the whole
    /// string parses and every referenced symbol is bound; otherwise reports the unbound symbols (or, for
    /// a malformed expression, the raw text) in `freeSymbols`. An empty expression is not resolvable here.
    inline DimEval evalDimExpr(const std::string &expr, const std::map<std::string, int64_t> &bindings) {
        DimEval out;
        if (expr.empty())
        {
            return out;
        }
        detail::DimExprParser p(expr, bindings, out.freeSymbols);
        int64_t               v = p.expr();
        p.skipws();
        if (p.bad || p.i != expr.size())
        {
            out.ok = false;
            if (out.freeSymbols.empty())
            {
                out.freeSymbols.push_back(expr); // name the unparseable expression for the diagnostic
            }
            return out;
        }
        out.ok    = out.freeSymbols.empty();
        out.value = v;
        return out;
    }

} // namespace vknn
