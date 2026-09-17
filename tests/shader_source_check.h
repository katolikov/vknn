// Host-side source checks that tie a test's C++ transcription of a GLSL kernel to the kernel's source text.
//
// GLSL never runs on the host, so a transcription is the only host proof of a shader's arithmetic, and it
// drifts silently when the .comp changes. These helpers read the shader and the C++ sources the kernel's
// interface lives in from the source tree (located from this header's own path: tests/*.cpp are globbed as
// absolute paths), strip comments, and compare what a test names: function bodies token for token after a
// normalization that removes only the spelling differences between the GLSL and its C++ transcription,
// named constants, push-constant members in order, binding declarations and the local size. A test skips
// when the sources are unreadable (a test binary run on a device).
#pragma once
#include <cctype>
#include <cstddef>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace shader_source {

    /// The repository root: the parent of the directory holding this header.
    inline std::string repositoryRoot() {
        const std::string header    = __FILE__;
        const size_t      separator = header.find_last_of("/\\");
        const std::string testsDir  = separator == std::string::npos ? std::string(".") : header.substr(0, separator);
        return testsDir + "/..";
    }

    /// The text of the repository file `relative` (e.g. "shaders/bitwise.comp"); empty when unreadable.
    inline std::string readRepositoryFile(const std::string &relative) {
        std::ifstream in(repositoryRoot() + "/" + relative, std::ios::binary);
        if (!in)
        {
            return {};
        }
        std::ostringstream text;
        text << in.rdbuf();
        return text.str();
    }

    /// `source` with every line comment and block comment removed (newlines kept) and every preprocessor
    /// line blanked (the checks compare declarations and code, never directives).
    inline std::string withoutCommentsAndDirectives(const std::string &source) {
        static constexpr size_t kCommentMarkerLength = 2;
        std::string             code;
        size_t                  at          = 0;
        bool                    lineHasCode = false;
        while (at < source.size())
        {
            if (source.compare(at, kCommentMarkerLength, "//") == 0)
            {
                at = source.find('\n', at);
                at = at == std::string::npos ? source.size() : at;
            } else if (source.compare(at, kCommentMarkerLength, "/*") == 0)
            {
                const size_t close = source.find("*/", at + kCommentMarkerLength);
                at                 = close == std::string::npos ? source.size() : close + kCommentMarkerLength;
            } else if (source[at] == '#' && !lineHasCode)
            {
                at = source.find('\n', at);
                at = at == std::string::npos ? source.size() : at;
            } else
            {
                lineHasCode = source[at] == '\n' ? false : (lineHasCode || !std::isspace((unsigned char) source[at]));
                code += source[at++];
            }
        }
        return code;
    }

    /// The tokens of C-family code, with the spellings that differ between GLSL and a C++ transcription
    /// normalized: `precise` is dropped, `constexpr` reads as `const`, `std::` qualifiers are dropped,
    /// `unsigned` and `uint32_t` read as `uint`, `int32_t` as `int`, a float literal loses its f suffix.
    inline std::vector<std::string> normalizedTokens(const std::string &code) {
        static constexpr const char *kTwoCharacterOperators[] = {
            "<=", ">=", "==", "!=", "&&", "||", "++", "--", "+=", "-=", "*=", "/=", "|=", "&=", "::", "<<", ">>"};
        static constexpr size_t kTwoCharacters = 2;
        auto                    wordCharacter  = [](char c) {
            return std::isalnum((unsigned char) c) || c == '_';
        };
        std::vector<std::string> tokens;
        size_t                   at = 0;
        while (at < code.size())
        {
            const char c = code[at];
            if (std::isspace((unsigned char) c))
            {
                ++at;
                continue;
            }
            size_t end = at + 1;
            if (wordCharacter(c))
            {
                const bool number = std::isdigit((unsigned char) c);
                while (end < code.size() && (wordCharacter(code[end]) || (number && code[end] == '.')))
                {
                    ++end;
                }
            } else
            {
                for (const char *twoCharacterOperator: kTwoCharacterOperators)
                {
                    if (code.compare(at, kTwoCharacters, twoCharacterOperator) == 0)
                    {
                        end = at + kTwoCharacters;
                        break;
                    }
                }
            }
            tokens.push_back(code.substr(at, end - at));
            at = end;
        }
        std::vector<std::string> normalized;
        for (size_t index = 0; index < tokens.size(); ++index)
        {
            std::string token = tokens[index];
            if (token == "precise")
            {
                continue;
            }
            if (token == "std" && index + 1 < tokens.size() && tokens[index + 1] == "::")
            {
                ++index;
                continue;
            }
            if (token == "constexpr")
            {
                token = "const";
            } else if (token == "unsigned" || token == "uint32_t")
            {
                token = "uint";
            } else if (token == "int32_t")
            { token = "int"; }
            if (std::isdigit((unsigned char) token[0]) && token.find('.') != std::string::npos && (token.back() == 'f' || token.back() == 'F'))
            {
                token.pop_back();
            }
            normalized.push_back(token);
        }
        return normalized;
    }

    namespace detail {

        // Copies the tokens of a parenthesized group starting at tokens[at] (an opening parenthesis) into
        // `out`; returns the index after its closing parenthesis.
        inline size_t copyParenthesized(const std::vector<std::string> &tokens, size_t at, std::vector<std::string> &out) {
            int depth = 0;
            do
            {
                depth += tokens[at] == "(" ? 1 : (tokens[at] == ")" ? -1 : 0);
                out.push_back(tokens[at++]);
            } while (at < tokens.size() && depth > 0);
            return at;
        }

        inline size_t copyStatement(const std::vector<std::string> &tokens, size_t at, std::vector<std::string> &out);

        // Copies the body of an if / else / for / while, enclosed in braces whether or not the source
        // braces it; returns the index after it.
        inline size_t copyBracedBody(const std::vector<std::string> &tokens, size_t at, std::vector<std::string> &out) {
            if (at < tokens.size() && tokens[at] == "{")
            {
                return copyStatement(tokens, at, out);
            }
            out.push_back("{");
            at = copyStatement(tokens, at, out);
            out.push_back("}");
            return at;
        }

        // Copies one statement starting at tokens[at]; returns the index after it.
        inline size_t copyStatement(const std::vector<std::string> &tokens, size_t at, std::vector<std::string> &out) {
            if (at >= tokens.size())
            {
                return at;
            }
            const std::string token = tokens[at];
            if (token == "{")
            {
                out.push_back(tokens[at++]);
                while (at < tokens.size() && tokens[at] != "}")
                {
                    at = copyStatement(tokens, at, out);
                }
                if (at < tokens.size())
                {
                    out.push_back(tokens[at++]);
                }
                return at;
            }
            if (token == "if" || token == "for" || token == "while")
            {
                out.push_back(tokens[at++]);
                if (at < tokens.size() && tokens[at] == "(")
                {
                    at = copyParenthesized(tokens, at, out);
                }
                at = copyBracedBody(tokens, at, out);
                if (token == "if" && at < tokens.size() && tokens[at] == "else")
                {
                    out.push_back(tokens[at++]);
                    at = copyBracedBody(tokens, at, out);
                }
                return at;
            }
            int depth = 0;
            while (at < tokens.size())
            {
                depth += tokens[at] == "(" ? 1 : (tokens[at] == ")" ? -1 : 0);
                const bool end = depth == 0 && tokens[at] == ";";
                out.push_back(tokens[at++]);
                if (end)
                {
                    break;
                }
            }
            return at;
        }

    } // namespace detail

    /// normalizedTokens of the braced block `block` (`{ ... }`) with every control-flow body braced: the form
    /// functionTokens gives a function body, for a test that states the shader body it expects.
    inline std::vector<std::string> normalizedBlock(const std::string &block) {
        std::vector<std::string> braced;
        detail::copyStatement(normalizedTokens(block), 0, braced);
        return braced;
    }

    /// A function definition's parameter list and body, as normalized tokens.
    struct FunctionTokens {
        std::vector<std::string> parameters; ///< from '(' through ')'
        std::vector<std::string> body;       ///< from '{' through '}', every control-flow body braced
        bool                     found = false;
    };

    /// The definition of function `name` in comment-free `code` (the first `name (` followed, after its
    /// parameter list, by a body). Control-flow bodies are braced in the result, so a GLSL `if (c) x;` and
    /// its C++ `if (c) { x; }` compare equal while the statement structure still counts.
    inline FunctionTokens functionTokens(const std::string &code, const std::string &name) {
        const std::vector<std::string> tokens = normalizedTokens(code);
        FunctionTokens                 function;
        for (size_t at = 0; at + 1 < tokens.size(); ++at)
        {
            if (tokens[at] != name || tokens[at + 1] != "(")
            {
                continue;
            }
            std::vector<std::string> parameters;
            const size_t             afterParameters = detail::copyParenthesized(tokens, at + 1, parameters);
            if (afterParameters >= tokens.size() || tokens[afterParameters] != "{")
            {
                continue; // a call or a declaration, not the definition
            }
            function.parameters = parameters;
            detail::copyStatement(tokens, afterParameters, function.body);
            function.found = true;
            return function;
        }
        return function;
    }

    /// The text of the braced block that follows the first occurrence of `opening` (e.g. "namespace glsl")
    /// in `code`, braces excluded; empty when absent.
    inline std::string blockAfter(const std::string &code, const std::string &opening) {
        const size_t start = code.find(opening);
        if (start == std::string::npos)
        {
            return {};
        }
        const size_t open  = code.find('{', start);
        int          depth = 0;
        for (size_t at = open; open != std::string::npos && at < code.size(); ++at)
        {
            depth += code[at] == '{' ? 1 : (code[at] == '}' ? -1 : 0);
            if (depth == 0)
            {
                return code.substr(open + 1, at - open - 1);
            }
        }
        return {};
    }

    /// Where constantDeclarations looks for declarations.
    enum class DeclarationScope { TopLevel, AnyDepth };

    /// Every `const <type> <name> = <value>;` declaration of comment-free `code` (C++ constexpr declarations
    /// included) at the top level, or at any brace depth, name -> "type = value" in normalized tokens.
    inline std::map<std::string, std::string> constantDeclarations(const std::string &code, DeclarationScope scope = DeclarationScope::TopLevel) {
        static constexpr size_t            kTypeOffset  = 1;
        static constexpr size_t            kNameOffset  = 2;
        static constexpr size_t            kEqualOffset = 3;
        std::map<std::string, std::string> constants;
        const std::vector<std::string>     tokens = normalizedTokens(code);
        int                                depth  = 0;
        for (size_t at = 0; at < tokens.size(); ++at)
        {
            depth += tokens[at] == "{" ? 1 : (tokens[at] == "}" ? -1 : 0);
            if ((scope == DeclarationScope::TopLevel && depth != 0) || tokens[at] != "const" || at + kEqualOffset >= tokens.size() || tokens[at + kEqualOffset] != "=")
            {
                continue;
            }
            std::string value = tokens[at + kTypeOffset] + " =";
            size_t      end   = at + kEqualOffset + 1;
            for (; end < tokens.size() && tokens[end] != ";"; ++end)
            {
                value += " " + tokens[end];
            }
            constants[tokens[at + kNameOffset]] = value;
            at                                  = end;
        }
        return constants;
    }

    /// The members of a `{ type a, b; type c; }` declaration block as "type name" in order.
    inline std::vector<std::string> declaredMembers(const std::string &block) {
        std::vector<std::string>       members;
        const std::vector<std::string> tokens = normalizedTokens(block);
        std::vector<std::string>       declaration;
        for (const std::string &token: tokens)
        {
            if (token != ";")
            {
                declaration.push_back(token);
                continue;
            }
            // type name (, name)* ;
            std::string type;
            size_t      at = 0;
            for (; at + 1 < declaration.size() && declaration[at + 1] != ","; ++at)
            {
                type += (type.empty() ? "" : " ") + declaration[at];
            }
            for (; at < declaration.size(); ++at)
            {
                if (declaration[at] != ",")
                {
                    members.push_back(type + " " + declaration[at]);
                }
            }
            declaration.clear();
        }
        return members;
    }

    /// The push-constant members of a comment-free shader, "type name" in order.
    inline std::vector<std::string> pushConstantMembers(const std::string &shaderCode) {
        return declaredMembers(blockAfter(shaderCode, "layout(push_constant)"));
    }

    /// The members of the first `struct <name>` in comment-free C++ `code`, "type name" in order.
    inline std::vector<std::string> structMembers(const std::string &code, const std::string &name) {
        return declaredMembers(blockAfter(code, "struct " + name + " "));
    }

    /// The binding indices of a comment-free shader's buffer declarations, in declaration order.
    inline std::vector<int> bindingIndices(const std::string &shaderCode) {
        static const std::string kBindingKey = "binding =";
        std::vector<int>         bindings;
        for (size_t at = shaderCode.find(kBindingKey); at != std::string::npos; at = shaderCode.find(kBindingKey, at + 1))
        {
            bindings.push_back(std::stoi(shaderCode.substr(at + kBindingKey.size())));
        }
        return bindings;
    }

    /// The local_size_x of a comment-free shader, -1 when it declares none.
    inline int localSizeX(const std::string &shaderCode) {
        static const std::string kLocalSizeKey = "local_size_x =";
        const size_t             at            = shaderCode.find(kLocalSizeKey);
        return at == std::string::npos ? -1 : std::stoi(shaderCode.substr(at + kLocalSizeKey.size()));
    }

    /// The value of the C++ integer constant `name` in comment-free `code` (`constexpr <type> name = value;`),
    /// or `fallback` when absent.
    inline long long cppIntegerConstant(const std::string &code, const std::string &name, long long fallback) {
        const std::map<std::string, std::string> constants = constantDeclarations(code, DeclarationScope::AnyDepth);
        const auto                               found     = constants.find(name);
        if (found == constants.end())
        {
            return fallback;
        }
        const size_t         equals                 = found->second.find('=');
        static constexpr int kLiteralBaseFromPrefix = 0;
        std::string          literal                = found->second.substr(equals + 2);
        while (!literal.empty() && (literal.back() == 'u' || literal.back() == 'U'))
        {
            literal.pop_back();
        }
        return std::stoll(literal, nullptr, kLiteralBaseFromPrefix);
    }

} // namespace shader_source
