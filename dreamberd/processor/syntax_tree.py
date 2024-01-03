from abc import ABCMeta
from dreamberd.base import Token, TokenType, InterpretationError
from dreamberd.processor.expression_tree import ExpressionNode, build_expression_tree

class CodeStatement(metaclass=ABCMeta):
    pass

class FunctionDefinition(CodeStatement):
    def __init__(self, name: str, code: list[CodeStatement]):
        self.name = name
        self.code = code

class ClassDeclaration(CodeStatement):
    def __init__(self, name: str, code: list[CodeStatement]):
        self.name = name
        self.code = code

class VariableAssignment(CodeStatement):
    def __init__(self, name: str, modifiers: list[str], lifetime: str, expression: ExpressionNode):
        self.name = name
        self.modifiers = modifiers
        self.lifetime = lifetime
        self.expression = expression

class ConditionalAssignment(CodeStatement):
    def __init__(self):
        pass

def split_into_statements(tokens: list[Token]) -> list[list[Token]]:
    statements = [[]]
    bracket_layers = 0
    for token in tokens:

        # check for expression-ending newlines
        if token.type == TokenType.WHITESPACE and not statements[-1]:  # don't care about whitespace at the beginning or end of an expression, idk
            continue
        if token.type == TokenType.NEWLINE and not statements[-1] and len(statements) > 1:
            statements[-2].append(token)
        else:
            statements[-1].append(token)

        # this is the start of a new scope, we don't care about those for RN
        if token.type == TokenType.L_CURLY:
            bracket_layers += 1 
        elif token.type == TokenType.R_CURLY:
            bracket_layers -= 1 

        if token.type in [TokenType.R_CURLY, TokenType.BANG, TokenType.QUESTION] and bracket_layers == 0:
            statements.append([])

    return statements

def generate_syntax_tree(filename: str, tokens: list[Token]): 
    """ Split the code up into lines, which are then parsed and shit """
    curr = 0
    current_names = {}
    while curr < len(tokens):
        token = tokens[curr]

        if token.type == TokenType.WHITESPACE:  # this is at the start of a line or statement
            if '\t' in token.value or len(token.value) % 3:
                raise InterpretationError(filename, token.line, "Indentation must be with spaces and a multiple of 3.")

        elif token.type in [TokenType.QUESTION, TokenType.BANG]:
            raise InterpretationError(filename, token.line, "Ending statement before creating it.")

        elif token.type != TokenType.NAME: 
            raise InterpretationError(filename, token.line, "Excepted a name or keyword at the beginning of a statement.")
