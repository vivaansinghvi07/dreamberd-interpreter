# basic things like the token class, keywords, etc.     
from __future__ import annotations

from enum import Enum
from typing import NoReturn, Optional
from dataclasses import dataclass, field

ALPH_NUMS = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.')

class NonFormattedError(Exception): pass

class InterpretationError(Exception):
    _           :(                                                                                              str)  # this is why i am the greatest programmer to ever live

def debug_print(filename: str, code: str, message: str, token: Token) -> None:
    if not code:  # adjust for repl-called code
        print(f'\n\033[33m{message}\033[39\n', sep="")
        return
    line = token.line
    num_carrots, num_spaces = len(token.value), token.col - len(token.value) + 1
    debug_string = f"\033[33m{filename}, line {line}\033[39m\n\n" + \
                   f"  {code.split(chr(10))[line - 1]}\n" + \
                   f" {num_spaces * ' '}{num_carrots * '^'}\n" + \
                   f"\033[33m{message}\033[39m"
    print('\n', debug_string, '\n', sep="")

def debug_print_no_token(filename: str, message: str) -> None:
    debug_string = f"\033[33m{filename}\033[39m\n\n" + \
                   f"\033[33m{message}\033[39m"
    print('\n', debug_string, '\n', sep="")

def raise_error_at_token(filename: str, code: str, message: str, token: Token) -> NoReturn:
    if not code:  # adjust for repl-called code
        raise InterpretationError(f"\n\033[31m{message}\033[39m\n")
    line = token.line
    num_carrots, num_spaces = len(token.value), token.col - len(token.value) + 1
    error_string = f"\033[33m{filename}, line {line}\033[39m\n\n" + \
                   f"  {code.split(chr(10))[line - 1]}\n" + \
                   f" {num_spaces * ' '}{num_carrots * '^'}\n" + \
                   f"\033[31m{message}\033[39m"
    raise InterpretationError(error_string)

def raise_error_at_line(filename: str, code: str, line: int, message: str) -> NoReturn:
    if not code:  # adjust for repl-called code
        raise InterpretationError(f"\n\033[31m{message}\033[39m\n")
    error_string = f"\033[33m{filename}, line {line}\033[39m\n\n" + \
                   f"  {code.split(chr(10))[line - 1]}\n\n" + \
                   f"\033[31m{message}\033[39m"
    raise InterpretationError(error_string)

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

    COMMA = ','
    COLON = ':'
    SEMICOLON = ';'
    BANG = '!'
    QUESTION = '?'
    CARROT = '^'
    FUNC_POINT = '=>'

    LESS_THAN = '<'
    GREATER_THAN = '>'
    LESS_EQUAL = '<='
    GREATER_EQUAL = '>='
    NOT_EQUAL = ';='  #!@#!@#!@#
    PIPE = '|'
    AND = '&'

    WHITESPACE = '       '
    NAME = 'abcaosdijawef'  # i'm losing my mind
    STRING = "'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''"  # iasddddakdjhnakjsndkjsbndfkijewbgf

    NEWLINE = '\n'
    SINGLE_QUOTE = "'"  # this is ugly as hell
    DOUBLE_QUOTE = '"'

    @classmethod
    def from_val(cls, val: str) -> Optional[TokenType]:
        return {v.value: v for v in list(cls)}.get(val)

class OperatorType(Enum):
    ADD = '+'
    SUB = '-'
    MUL = '*'
    DIV = '/'
    EXP = '^'
    GT  = '>'
    GE  = '>='
    LT  = '<'
    LE  = '<='
    OR  = '|'
    AND = '&'
    COM = ','  # this is just here to seperate variables in a function 
    E   = '='
    EE  = '=='
    EEE = '==='
    EEEE= '===='
    NE  = ';='
    NEE = ';=='
    NEEE= ';==='

STR_TO_OPERATOR = {op.value: op for op in OperatorType}

# why do i even need a damn class for this 
# 3 weeks later, i am very glad i made a class for this
@dataclass(unsafe_hash=True)
class Token():

    type: TokenType
    value: str 
    line: int = field(hash=False)
    col: int = field(hash=False)

    def __repr__(self) -> str:
        return f"Token({self.type}, {repr(self.value)})"
