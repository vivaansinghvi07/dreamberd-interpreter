from __future__ import annotations
from typing import Optional 

from dreamberd.base import Token, TokenType, ALPH_NUMS, raise_error_at_line

# thanks : https://craftinginterpreters.com/scanning.html

def add_to_tokens(token_list: list[Token], line: int, col: int, token: TokenType, value: Optional[str] = None):
    token_list.append(Token(token, value if value is not None else token.value, line, col))

def get_effective_whitespace_value(char: str) -> str:
    match char:
        case " " | "(":
            return " "
        case "\t":
            return char 
    return ""
    
def get_quote_count(quote_value: str) -> int:
    return sum(2 if c == '"' else 1 for c in quote_value)

def is_matching_pair(quote_value: str) -> bool:
    """ 
    Finds a pair of quote groups that have the same count of quotes. 
    Returns an integer index where the second group begins if found, else -1.
    """
    total_sum = get_quote_count(quote_value)
    if total_sum % 2: return False
    for i in range(len(quote_value)):
        if get_quote_count(quote_value[:i]) == total_sum // 2:
            return True
    return False

def get_string_token(code: str, curr: int, filename: str, error_line: int) -> tuple[int, str]:

    """ 
    Scans the code for the shortest possible string and returns it. 
    Returns as soon as a pair of quote groups is found that is equal in terms of quote count on both sides.
    For example, """""" reads the two first double quotes, detects that there is a pair (" and "), and returns the corresponding empty string.
    To have more sequences of quotes, one can do the following:
        '""hello world"'"  <-- this is interpreted as the string "hello world"
    Therefore, to avoid premature returns of quotes, simply preface your quotes with a single ' and the rest "
    This guarantees that no pair of quotes will be found in the starting quote because it will have an odd number of quotes.
    """

    quote_value = ""
    while code[curr] in """"'""":  # lmaoo
        quote_value += code[curr] 
        if is_matching_pair(quote_value):
            return curr, ""
        curr += 1
    quote_count = get_quote_count(quote_value)

    value = ""
    while curr < len(code):
        running_count, quote_start = 0, curr
        while code[curr] in """"'""":
            running_count += 2 if code[curr] == '"' else 1
            if running_count == quote_count:
                return curr, value
            curr += 1
        value += code[quote_start:curr + 1]
        curr += 1
    else:
        raise_error_at_line(filename, code, error_line, "Invalid string. Starting quotes do not match opening quotes.")

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
                curr, value = get_string_token(code, curr, filename, line_count)
                add_to_tokens(tokens, line_count, curr - start, TokenType.STRING, value)
            case ' ' | '\t' | '(' | ')': 
                if code[curr] == '(' and curr + 1 < len(code) and code[curr + 1] == ')':
                    add_to_tokens(tokens, line_count, curr - start, TokenType.WHITESPACE, '')
                    add_to_tokens(tokens, line_count, curr - start, TokenType.NAME, '')  # please please please work
                    add_to_tokens(tokens, line_count, curr - start, TokenType.WHITESPACE, '')
                    curr += 1
                else:
                    value = get_effective_whitespace_value(code[curr])
                    while curr + 1 < len(code) and code[curr + 1] in ' ()\t':
                        value += get_effective_whitespace_value(code[curr + 1])
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
