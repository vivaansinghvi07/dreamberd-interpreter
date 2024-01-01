from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import Optional

from base import STR_TO_OPERATOR, Token, TokenType, OperatorType, InterpretationError, VAR_DECL_KW, raise_error_at_line, raise_error_at_token

class SyntaxTreeNode(metaclass=ABCMeta):
    @abstractmethod
    def to_string(self, tabs: int = 0) -> str: pass

class ListNode(SyntaxTreeNode):
    def __init__(self, values: list[SyntaxTreeNode]):
        self.values = values
    def to_string(self, tabs: int = 0) -> str:
        return f"{'  ' * tabs}List: \n" + \
               f"{'  ' * (tabs + 1)}Values: \n" + \
               "\n".join([f"{v.to_string(tabs + 2)}" for v in self.values])

class ExpressionNode(SyntaxTreeNode):
    def __init__(self, left: SyntaxTreeNode, right: SyntaxTreeNode, operator: OperatorType):
        self.left = left 
        self.right = right
        self.operator = operator
    def to_string(self, tabs: int = 0) -> str:
        return f"{'  ' * tabs}Expression: \n" + \
               f"{'  ' * (tabs + 1)}Operator: {self.operator}\n" + \
               f"{'  ' * (tabs + 1)}Left: \n" + \
               f"{self.left.to_string(tabs + 2)}\n" + \
               f"{'  ' * (tabs + 1)}Right: \n" + \
               f"{self.right.to_string(tabs + 2)}"

class FunctionNode(SyntaxTreeNode):
    def __init__(self, name: str, args: list[SyntaxTreeNode]):
        self.name = name 
        self.args = args
    def to_string(self, tabs: int = 0) -> str:
        return f"{'  ' * tabs}Function: \n" + \
               f"{'  ' * (tabs + 1)}Name: {self.name}\n" + \
               f"{'  ' * (tabs + 1)}Arguments: \n" + \
               "\n".join([f"{arg.to_string(tabs + 2)}" for arg in self.args]) 

class Value(SyntaxTreeNode):
    def __init__(self, name_or_value: str, index: Optional[SyntaxTreeNode] = None): 
        self.name_or_value = name_or_value
        self.index = index
    def to_string(self, tabs: int = 0) -> str:
        return f"{'  ' * tabs}Value: {self.name_or_value}"

def build_expression_tree(filename: str, tokens: list[Token], code: str) -> SyntaxTreeNode:
    """ 
    This language has significant whitespace, so the biggest split happens where there is most space
     - func a, b  +  c becomes func(a, b) + c but func a, b+c  becomes func(a, b + c) 
     - a + func  b  ,  c + d is not legal because it translates to (a + func)(b, c + d)
     - 2 * 1+3 becomes 2 * (1 + 3)
    """

    for token in tokens:
        if token.type == TokenType.WHITESPACE and '\t' in token.value:
            raise_error_at_token(filename, code, "Tabs are not allowed in expressions.", token)
    
    # create a new list consisting and tokens and a brand new type: the list 
    tokens_without_whitespace = [token for token in tokens if token.type != TokenType.WHITESPACE]

    # transform a list of tokens to include operators 
    # find the operator with the maximum whitespace between it and other things
    updated_list = [STR_TO_OPERATOR.get(token.value, token) for token in tokens]
    max_width, max_index = 0, -1
    for i in range(len(updated_list)):
        if isinstance(updated_list[i], OperatorType):
            try:
                l_len, r_len = 0, 0
                if tokens[i - 1].type == TokenType.WHITESPACE:
                    l_len = len(tokens[i - 1].value)
                if tokens[i + 1].type == TokenType.WHITESPACE:
                    r_len = len(tokens[i + 1].value)
                if l_len != r_len and updated_list[i] != OperatorType.COM:
                    raise_error_at_token(filename, code, "Whitespace must be equal on either side of an operator.", tokens[i])
                if r_len >= max_width:
                    max_width = r_len
                    max_index = i
            except IndexError:
                raise_error_at_token(filename, code, "Operator cannot be at the end of an expression.", tokens[i])

    # there is no operator, must be just a value
    if max_index == -1:
        name_or_value = tokens_without_whitespace[0]
        if name_or_value.type != TokenType.NAME:
            raise_error_at_token(filename, code, "Expected name or value.", tokens_without_whitespace[0])
        end, start = 0, 0  # this is just here so pyright won't yell at me
        if any(l := [token.type == TokenType.L_SQUARE for token in tokens]):
            start, end = l.index(True), len(tokens) - 1
            while end > start:
                if tokens[end] == TokenType.R_SQUARE:
                    break
                end -= 1
            else:
                raise_error_at_line(filename, code, tokens[start].line, "Excepted closing brace for index operation.")
        return Value(name_or_value.value, build_expression_tree(filename, tokens[start + 1 : end], code) if any(l) else None)  
        
    # max_index is the token with the maximum surrouding whitespace 
    if updated_list[max_index].value == ',':  
        # this means it is a function
        # we need to find every other comma as they become the arguments of the function
        # additionally, there needs to be a spacing of equal length between the name of the function and the next argument
        
        is_valid_list = tokens_without_whitespace[0].type == TokenType.L_SQUARE
        is_valid_func = tokens_without_whitespace[0].type == TokenType.NAME and tokens_without_whitespace[1].type == TokenType.NAME
        if not is_valid_list and not is_valid_func:
            raise_error_at_token(filename, code, "Expected function call. This is likely an issue of whitespace, as DreamBerd replaces parentheses with spaces and has significant whitespace.", tokens_without_whitespace[0])
        
        all_commas = []
        for i in range(len(updated_list)):
            if updated_list[i].value == ',':
                if max_width == 0 or (tokens[i + 1].type == TokenType.WHITESPACE and len(tokens[i + 1].value)) == max_width:
                    all_commas.append(i)
        
        # now can split expressions within and then call the function
        if is_valid_func:
            return FunctionNode(tokens_without_whitespace[0].value, [
                build_expression_tree(filename, t, code) for t in [
                    tokens[comma_index + 1:next_index] for comma_index, next_index in 
                    zip([int(tokens[0].type == TokenType.WHITESPACE), *all_commas], [*all_commas, len(tokens)])
                ]
            ])
        elif is_valid_list:
            print(all_commas)
            print(tokens)
            return ListNode([
                build_expression_tree(filename, t, code) for t in [
                    tokens[comma_index + 1:next_index] for comma_index, next_index in 
                    zip([int(tokens[0].type == TokenType.WHITESPACE), *all_commas], 
                        [*all_commas, len(tokens) - 1 - int(tokens[-1].type == TokenType.WHITESPACE)])  # adjusting here in order to avoid the bracket tokens
                ]
            ])

    else: 
        return ExpressionNode(
            build_expression_tree(filename, tokens[:max_index], code), 
            build_expression_tree(filename, tokens[max_index + 1:], code), 
            operator=updated_list[max_index]
        )

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

        if token.value in VAR_DECL_KW:
            modifier = []
            
            while token.value in VAR_DECL_KW or token.type == TokenType.WHITESPACE:
                if token.value in VAR_DECL_KW:
                    modifier.append(token.value)
                curr += 1
                token = tokens[curr]          
        
            names = []
            lifetime = ''
            while token.type != TokenType.EQUAL:

                # apparently splitting in this does nothing??
                if token.type in {TokenType.L_SQUARE, TokenType.R_SQUARE, TokenType.COMMA, TokenType.WHITESPACE}:
                    continue

                # lifetime detected in the form of <something>
                elif token.type == TokenType.LESS_THAN:
                    while token.type != TokenType.GREATER_THAN:  # looking for closing >
                        if token.type == TokenType.WHITESPACE:
                            continue
                        elif token.type != TokenType.NAME:
                            raise InterpretationError(filename, token.line, "Excepted name in lifetime declaration.")
                        lifetime = token.value
                        curr += 1
                        token = tokens[curr] 
                    curr += 1  
                    token = tokens[curr]  # skip over the greater than token
                    continue

                # anything not a name is a bad token
                elif token.type != TokenType.NAME:
                    raise InterpretationError(filename, token.line, "Expected a name for variable declaration.")
                names.append(token.value)
                curr += 1
                token = tokens[curr]

            # add to all the names that are naming
            current_names |= set(names) 

            if token.value != '=':
                raise InterpretationError(filename, token.line, "Expected a single equal sign for assignment.")
                    
            expression_tokens = []
            while token.type not in [TokenType.BANG, TokenType.QUESTION]:  # evaluate the expression till the end of the line
                expression_tokens.append(token)
                curr += 1
                token = tokens[curr]
    
            expression_root = build_expression_tree(expression_tokens)
            # now it's either a bang or question mark

