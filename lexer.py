import re
from enum import Enum
from typing import Optional 
from dataclasses import dataclass

# thanks : https://craftinginterpreters.com/scanning.html

KEYWORDS = ['print', ]
ALH_NUMS = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_')

class InterpretationError(Exception):
    _           :(                                                                                              str)  # this is why i am the greatest programmer to ever live
   
class TokenType(Enum): 
    R_CURLY = '}'
    L_CURLY = '{'
    R_SQUARE = ']'
    L_SQUARE = '['

    DOT = '.'
    ADD = '+'
    INCREMENT = '++'
    DECREMENT = '--'
    EQUAL = '='
    DIVIDE = '/'
    MULTIPLY = '*'
    SUBTRACT = '-'

    SEMICOLON = ';'
    BANG = '!'
    QUESTION = '?'
    CARROT = '^'
    FUNC_POINT = '=>'

    WHITESPACE = '       '
    NAME = 'abcaosdijawef'  # i'm losing my mind
    STRING = "'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''"  # iasddddakdjhnakjsndkjsbndfkijewbgf

    NEWLINE = '\n'
    SINGLE_QUOTE = "'"  # this is ugly as hell
    DOUBLE_QUOTE = '"'

# why do i even need a damn class for this 
@dataclass
class Token():
    def __init__(self, type: TokenType, value: str) -> None:
        self.type = type
        self.value = value
    def __repr__(self) -> str:
        return f"Token({self.type}, {repr(self.value)})"

# oh okay we're doing OOP now, nvm,
class TokenList():
    def __init__(self):
        self.tokens = []

    def add(self, token: TokenType, value: Optional[str] = None):
        self.tokens.append(Token(token, value or token.value))

    def __str__(self) -> str:
        return str(self.tokens)

def tokenize(code: str) -> None:
    code += ' '  # adding a space here so i dont have to write 10 damn checks for out of bounds
    line_count = 0
    tokens = TokenList()
    curr = 0
    while curr < len(code):
        match code[curr]:
            case '(' | ')': pass   # apparently this stupid f*cking language doesn't use these (sorry i didnt mean it)
            case '\n':
                line_count += 1
                tokens.add(TokenType.NEWLINE)
            case '}': tokens.add(TokenType.R_CURLY)
            case '{': tokens.add(TokenType.L_CURLY)
            case '[': tokens.add(TokenType.L_SQUARE)
            case ']': tokens.add(TokenType.R_SQUARE)
            case '.': tokens.add(TokenType.DOT)
            case ';': tokens.add(TokenType.SEMICOLON)
            case '+': 
                if code[curr + 1] == '+':
                    tokens.add(TokenType.INCREMENT)
                    curr += 1
                else:
                    tokens.add(TokenType.ADD)  # YOU NEVER SAID I HAD TO DO +=
            case '-': 
                if code[curr + 1] == '-':
                    tokens.add(TokenType.DECREMENT)
                    curr += 1
                else:
                    tokens.add(TokenType.SUBTRACT)
            case '*': tokens.add(TokenType.MULTIPLY)
            case '/': tokens.add(TokenType.DIVIDE)
            case '^': tokens.add(TokenType.CARROT)
            case '!':
                value = '!'
                while code[curr + 1] == '!':
                    value += '!'
                    curr += 1
                tokens.add(TokenType.BANG, value)
            case '?': 
                if curr < len(code) - 1 and code[curr + 1] == '?':
                    raise InterpretationError(f"User is too confused on line {line_count}. Aborting due to trust issues.")  # heheheheheheh
                tokens.add(TokenType.QUESTION)
            case '=': 
                value = '='
                if code[curr + 1] == '>':
                    curr += 1 
                    tokens.add(TokenType.FUNC_POINT)
                else: 
                    while code[curr + 1] == '=':
                        value += '='
                        curr += 1
                    tokens.add(TokenType.EQUAL, value)
            case '"' | "'":
                quote_count = 0
                while code[curr] in "'\"":
                    quote_count += 1 if code[curr] == "'" else 2
                    curr += 1
                value = ''
                while code[curr] not in "'\"":
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
                        raise InterpretationError("Invalid string. Starting parentheses do not match opening parentheses")
                curr -= 1
                tokens.add(TokenType.STRING, value)
            case ' ' | '\t':
                value = code[curr]
                while curr + 1 < len(code) and code[curr + 1] in ' \t':
                    curr += 1
                    value += code[curr]
                tokens.add(TokenType.WHITESPACE, value)
            case c:
                value = c
                while code[curr + 1] in ALH_NUMS:
                    curr += 1
                    value += code[curr]
                tokens.add(TokenType.NAME, value)
                
        curr += 1

    print(tokens)
