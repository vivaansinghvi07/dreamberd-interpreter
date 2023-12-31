# basic things like the token class, keywords, etc.     

from enum import Enum
from dataclasses import dataclass

ALPH_NUMS = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_')
VAR_DECL_KW = {'const', 'var'}

class InterpretationError(Exception):
    _           :(                                                                                              str)  # this is why i am the greatest programmer to ever live
    
    def __init__(self, file: str, line: int, message: str) -> None:
        self.message = "\033[33m" + f"{file}, line {line}\n  " + "\033[31m" + message + "\033[39m"

    def __str__(self):
        return self.message
   
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
    SEMICOLON = ';'
    BANG = '!'
    QUESTION = '?'
    CARROT = '^'
    FUNC_POINT = '=>'

    LESS_THAN = '<'
    GREATER_THAN = '>'
    LESS_EQUAL = '<='
    GREATER_EQUAL = '>='

    WHITESPACE = '       '
    NAME = 'abcaosdijawef'  # i'm losing my mind
    STRING = "'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''"  # iasddddakdjhnakjsndkjsbndfkijewbgf

    NEWLINE = '\n'
    SINGLE_QUOTE = "'"  # this is ugly as hell
    DOUBLE_QUOTE = '"'

class OperatorType(Enum):
    ADD = '+'
    SUB = '-'
    MUL = '*'
    DIV = '/'
    EXP = '^'
    GT  = '>'
    GE  = '>='
    LT  = '<'
    LE  = '>'
    OR  = '||'
    AND = '&&'
    COM = ','  # this is just here to seperate variables in a function 

STR_TO_OPERATOR = {op.value: op for op in OperatorType}

# why do i even need a damn class for this 
@dataclass
class Token():
    def __init__(self, type: TokenType, value: str, line: int) -> None:
        self.type = type
        self.value = value
        self.line = line
    def __repr__(self) -> str:
        return f"Token({self.type}, {repr(self.value)})"


