from __future__ import annotations
from typing import Optional 

from dreamberd.base import Token, TokenType, ALPH_NUMS, raise_error_at_line

# thanks : https://craftinginterpreters.com/scanning.html

def add_to_tokens(token_list: list[Token], line: int, col: int, token: TokenType, value: Optional[str] = None):
    token_list.append(Token(token, value if value is not None else token.value, line, col))

def tokenize(filename: str, code: str) -> list[Token]:
    code += '   '  # adding a space here so i dont have to write 10 damn checks for out of bounds
    line_count = 1
    tokens = []
    curr, start = 0, 0
    while curr < len(code):
        match code[curr]:
            case '\n':
                line_count += 1
                start = curr  # at the new line to get col number
                add_to_tokens(tokens, line_count, curr - start, TokenType.NEWLINE)
            case '}': add_to_tokens(tokens, line_count, curr - start, TokenType.R_CURLY)
            case '{': add_to_tokens(tokens, line_count, curr - start, TokenType.L_CURLY)
            case '[': add_to_tokens(tokens, line_count, curr - start, TokenType.L_SQUARE)
            case ']': add_to_tokens(tokens, line_count, curr - start, TokenType.R_SQUARE)
            case '.': add_to_tokens(tokens, line_count, curr - start, TokenType.DOT)
            case ':': add_to_tokens(tokens, line_count, curr - start, TokenType.COLON)
            case '|': add_to_tokens(tokens, line_count, curr - start, TokenType.PIPE)
            case '&': add_to_tokens(tokens, line_count, curr - start, TokenType.AND)
            case ';': 
                value = ';'
                while code[curr + 1] == '=':
                    value += '='
                    curr += 1
                if len(value) == 1:
                    add_to_tokens(tokens, line_count, curr - start, TokenType.SEMICOLON)
                else:
                    add_to_tokens(tokens, line_count, curr - start, TokenType.NOT_EQUAL, value)
            case ',': add_to_tokens(tokens, line_count, curr - start, TokenType.COMMA)
            case '+': 
                if code[curr + 1] == '+':
                    add_to_tokens(tokens, line_count, curr - start, TokenType.INCREMENT)
                    curr += 1
                else:
                    add_to_tokens(tokens, line_count, curr - start, TokenType.ADD)  # YOU NEVER SAID I HAD TO DO +=
            case '-': 
                if code[curr + 1] == '-':
                    add_to_tokens(tokens, line_count, curr - start, TokenType.DECREMENT)
                    curr += 1
                else:
                    add_to_tokens(tokens, line_count, curr - start, TokenType.SUBTRACT)
            case '*': add_to_tokens(tokens, line_count, curr - start, TokenType.MULTIPLY)
            case '/': add_to_tokens(tokens, line_count, curr - start, TokenType.DIVIDE)
            case '^': add_to_tokens(tokens, line_count, curr - start, TokenType.CARROT)
            case '>': 
                if code[curr + 1] == '=':
                    add_to_tokens(tokens, line_count, curr - start, TokenType.GREATER_EQUAL)
                    curr += 1
                else:
                    add_to_tokens(tokens, line_count, curr - start, TokenType.GREATER_THAN)
            case '<': 
                if code[curr + 1] == '=':
                    add_to_tokens(tokens, line_count, curr - start, TokenType.LESS_EQUAL)
                    curr += 1
                else:
                    add_to_tokens(tokens, line_count, curr - start, TokenType.LESS_THAN)
            case '!':
                value = '!'
                while code[curr + 1] == '!':
                    value += '!'
                    curr += 1
                add_to_tokens(tokens, line_count, curr - start, TokenType.BANG, value)
            case '?': 
                value = '?'
                while code[curr + 1] == '?':
                    value += '?'
                    curr += 1
                if len(value) > 4:
                    raise_error_at_line(filename, code, line_count, "User is too confused. Aborting due to trust issues.")  # heheheheheheh
                add_to_tokens(tokens, line_count, curr - start, TokenType.QUESTION, value)
            case '=': 
                value = '='
                if code[curr + 1] == '>':
                    curr += 1 
                    add_to_tokens(tokens, line_count, curr - start, TokenType.FUNC_POINT)
                else: 
                    while code[curr + 1] == '=':
                        value += '='
                        curr += 1
                    add_to_tokens(tokens, line_count, curr - start, TokenType.EQUAL, value)
            case '"' | "'":
                quote_count = 0
                while code[curr] in "'\"":
                    quote_count += 1 if code[curr] == "'" else 2
                    curr += 1
                value = ''
                while code[curr] not in "'\"\n":   # changing this here because \n not allowed in strings
                    value += code[curr]
                    if code[curr] == "\\":
                        curr += 1
                        value += code[curr]
                    curr += 1
                while code[curr] in "'\"":
                    quote_count -= 1 if code[curr] == "'" else 2
                    curr += 1
                if code[curr] not in '!\n':  # autocomplete of strings.......................
                    if quote_count != 0:
                        raise_error_at_line(filename, code, line_count, "Invalid string. Starting quotes do not match opening quotes.")
                curr -= 1
                add_to_tokens(tokens, line_count, curr - start, TokenType.STRING, value)
            case ' ' | '\t' | '(' | ')': 
                if code[curr] == '(' and curr + 1 < len(code) and code[curr + 1] == ')':
                    add_to_tokens(tokens, line_count, curr - start, TokenType.WHITESPACE, '')
                    add_to_tokens(tokens, line_count, curr - start, TokenType.NAME, '')  # please please please work
                    add_to_tokens(tokens, line_count, curr - start, TokenType.WHITESPACE, '')
                    curr += 1
                else:
                    value = code[curr] if code[curr] not in '()' else ' '
                    while curr + 1 < len(code) and code[curr + 1] in ' ()\t':
                        if code[curr + 1] not in "()":
                            value += code[curr + 1]
                        curr += 1
                    add_to_tokens(tokens, line_count, curr - start, TokenType.WHITESPACE, value)
            case c:
                value = c
                while code[curr + 1] in ALPH_NUMS:
                    curr += 1
                    value += code[curr]
                add_to_tokens(tokens, line_count, curr - start, TokenType.NAME, value)
        curr += 1
    return tokens
