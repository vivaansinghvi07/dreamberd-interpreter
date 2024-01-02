from __future__ import annotations
from abc import ABCMeta, abstractmethod

from base import STR_TO_OPERATOR, Token, TokenType, OperatorType, InterpretationError, VAR_DECL_KW, raise_error_at_token

class ExpressionTreeNode(metaclass=ABCMeta):
    @abstractmethod
    def to_string(self, tabs: int = 0) -> str: pass

class ListNode(ExpressionTreeNode):
    def __init__(self, values: list[ExpressionTreeNode]):
        self.values = values
    def to_string(self, tabs: int = 0) -> str:
        return f"{'  ' * tabs}List: \n" + \
               f"{'  ' * (tabs + 1)}Values: \n" + \
               "\n".join([f"{v.to_string(tabs + 2)}" for v in self.values])

class ExpressionNode(ExpressionTreeNode):
    def __init__(self, left: ExpressionTreeNode, right: ExpressionTreeNode, operator: OperatorType):
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

class FunctionNode(ExpressionTreeNode):
    def __init__(self, name: str, args: list[ExpressionTreeNode]):
        self.name = name 
        self.args = args
    def to_string(self, tabs: int = 0) -> str:
        return f"{'  ' * tabs}Function: \n" + \
               f"{'  ' * (tabs + 1)}Name: {self.name}\n" + \
               f"{'  ' * (tabs + 1)}Arguments: \n" + \
               "\n".join([f"{arg.to_string(tabs + 2)}" for arg in self.args]) 

class IndexNode(ExpressionTreeNode):
    def __init__(self, value: ExpressionTreeNode, index: ExpressionTreeNode):
        self.value = value 
        self.index = index
    def to_string(self, tabs: int = 0) -> str:
        return f"{'  ' * tabs}Index: \n" + \
               f"{'  ' * (tabs + 1)}Of: \n" + \
               f"{self.value.to_string(tabs + 2)}\n" + \
               f"{'  ' * (tabs + 1)}At: \n" + \
               f"{self.index.to_string(tabs + 2)}"

class Value(ExpressionTreeNode):
    def __init__(self, name_or_value: Token): 
        self.name_or_value = name_or_value
    def to_string(self, tabs: int = 0) -> str:
        return f"{'  ' * tabs}Value: {self.name_or_value}"

def build_expression_tree(filename: str, tokens: list[Token], code: str) -> ExpressionTreeNode:
    print(tokens)
    """ 
    This language has significant whitespace, so the biggest split happens where there is most space
     - func a, b  +  c becomes func(a, b) + c but func a, b+c  becomes func(a, b + c) 
     - a + func  b  ,  c + d is not legal because it translates to (a + func)(b, c + d)
     - 2 * 1+3 becomes 2 * (1 + 3)

    TODO: might need to refactor this to not consider commas within brackets at all.
          this way list parsing can be done within the "if max_width == -1" thing
    """

    for token in tokens:
        if token.type == TokenType.WHITESPACE and '\t' in token.value:
            raise_error_at_token(filename, code, "Tabs are not allowed in expressions.", token)
    
    # create a new list consisting and tokens and a brand new type: the list 
    tokens_without_whitespace = [token for token in tokens if token.type != TokenType.WHITESPACE]
    starts_with_whitespace = tokens[0].type == TokenType.WHITESPACE
    ends_with_whitespace = tokens[-1].type == TokenType.WHITESPACE

    # transform a list of tokens to include operators 
    # find the operator with the maximum whitespace between it and other things
    updated_list = [STR_TO_OPERATOR.get(token.value, token) for token in tokens]
    max_width, max_index = 0, -1
    bracket_layers = 0
    for i in range(len(updated_list)):
        if tokens[i].type == TokenType.L_SQUARE:
            bracket_layers += 1
        elif tokens[i].type == TokenType.R_SQUARE:
            bracket_layers -= 1
        if isinstance(updated_list[i], OperatorType) and bracket_layers == 0 or updated_list[i] == OperatorType.COM and bracket_layers == 1:
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

    # detecting single argument function
    # this doesn't seem to adhere to my standards 100%, so its not a bug, its a feature
    first_name_index = int(starts_with_whitespace)
    if len(tokens) >= 3 + first_name_index and \
       tokens[first_name_index].type == TokenType.NAME and \
       tokens[first_name_index + 1].type == TokenType.WHITESPACE and \
       tokens[first_name_index + 2].type in [TokenType.NAME, TokenType.L_SQUARE, TokenType.STRING] and \
       len(tokens[first_name_index + 1].value) > max_width:
        return FunctionNode(tokens[first_name_index].value,
                            [build_expression_tree(filename, tokens[first_name_index + 1:], code)])

    # there is no operator, must be just a value
    if max_index == -1:
        try:
            name_or_value = tokens_without_whitespace[0]
            if name_or_value.type not in [TokenType.NAME, TokenType.L_SQUARE, TokenType.STRING]:
                raise_error_at_token(filename, code, "Expected name or value.", tokens_without_whitespace[0])
        except IndexError:
            raise_error_at_token(filename, code, "Expected name or value.", tokens_without_whitespace[0])

        # detecting single element list (why??? nobody does this)
        # okay we need to see when the list ends and consider that as the "list" we also need to check if this damn thing is being indexed
        if name_or_value.type == TokenType.L_SQUARE:
            bracket_layers = 1
            for i, token in enumerate(tokens_without_whitespace[1:], start=1):
                if token.type == TokenType.L_SQUARE:
                    bracket_layers += 1 
                elif token.type == TokenType.R_SQUARE:
                    bracket_layers -= 1

                # this means there is no index because the closing is happening late
                if bracket_layers == 0:
                    if i == len(tokens_without_whitespace) - 1:
                        return ListNode([build_expression_tree(filename, tokens[int(starts_with_whitespace) + 1 : len(tokens) - int(ends_with_whitespace) - 1], code)])
                    break

        # now we need to handle indexes
        # let's go from the back of the list and find the first fully closing sequence 
        # i am sure that this guarantees there is an index (i think)
        if tokens_without_whitespace[-1].type == TokenType.R_SQUARE:
            bracket_layers = -1
            end_index = len(tokens) - int(ends_with_whitespace) - 1
            for i, token in reversed(list(enumerate(tokens[:end_index]))):  # i don't like this one bit  :(
                if token.type == TokenType.L_SQUARE:
                    bracket_layers += 1 
                elif token.type == TokenType.R_SQUARE:
                    bracket_layers -= 1

                # first index!!!!!!!!!!!!!!!!!!!!
                if bracket_layers == 0:
                    return IndexNode(build_expression_tree(filename, tokens[int(starts_with_whitespace) : i], code),
                                     build_expression_tree(filename, tokens[i + 1 : end_index], code))
                    
        return Value(name_or_value)
        
    # max_index is the token with the maximum surrouding whitespace 
    if updated_list[max_index].value == ',':  
        # this means it is a function
        # we need to find every other comma as they become the arguments of the function
        # additionally, there needs to be a spacing of equal length between the name of the function and the next argument
        
        is_valid_list = tokens_without_whitespace[0].type == TokenType.L_SQUARE
        is_valid_func = tokens_without_whitespace[0].type == TokenType.NAME and tokens_without_whitespace[1].type in [TokenType.NAME, TokenType.L_SQUARE, TokenType.STRING] 
        if not is_valid_list and not is_valid_func:
            raise_error_at_token(filename, code, "Expected function call. This is likely an issue of whitespace, as DreamBerd replaces parentheses with spaces and has significant whitespace.", tokens_without_whitespace[0])
        
        all_commas = []
        for i in range(len(updated_list)):
            if updated_list[i].value == ',':  # okay this is weird because enums have .value and tokens have .value
                if max_width == 0 or (tokens[i + 1].type == TokenType.WHITESPACE and len(tokens[i + 1].value)) == max_width:
                    all_commas.append(i)
        
        # now can split expressions within and then call the function
        if is_valid_func:
            bracket_layers = 0
            for i, token in enumerate(tokens):
                if token.type == TokenType.L_SQUARE:
                    bracket_layers += 1 
                elif token.type == TokenType.R_SQUARE:
                    bracket_layers -= 1
                if bracket_layers != 0 and i in all_commas:
                    return FunctionNode(tokens_without_whitespace[0].value, [build_expression_tree(filename, tokens[int(starts_with_whitespace) + 1:], code)])

            return FunctionNode(tokens_without_whitespace[0].value, [
                build_expression_tree(filename, t, code) for t in [
                    tokens[comma_index + 1:next_index] for comma_index, next_index in 
                    zip([int(starts_with_whitespace), *all_commas], [*all_commas, len(tokens)])
                ]
            ])
        elif is_valid_list:

            bracket_layers = -1
            end_index = len(tokens) - int(ends_with_whitespace) - 1
            for i, token in reversed(list(enumerate(tokens[:end_index]))):  # i don't like this one bit  :(
                if token.type == TokenType.L_SQUARE:
                    bracket_layers += 1 
                elif token.type == TokenType.R_SQUARE:
                    bracket_layers -= 1

                # first index!!!!!!!!!!!!!!!!!!!!
                if bracket_layers == 0 and i > int(starts_with_whitespace):
                    return IndexNode(build_expression_tree(filename, tokens[int(starts_with_whitespace) : i], code),
                                     build_expression_tree(filename, tokens[i + 1 : end_index], code))

            return ListNode([
                build_expression_tree(filename, t, code) for t in [
                    tokens[comma_index + 1:next_index] for comma_index, next_index in 
                    zip([int(starts_with_whitespace), *all_commas], 
                        [*all_commas, len(tokens) - 1 - int(ends_with_whitespace)])  # adjusting here in order to avoid the bracket tokens
                ]
            ])

    else: 
        return ExpressionNode(
            build_expression_tree(filename, tokens[:max_index], code), 
            build_expression_tree(filename, tokens[max_index + 1:], code), 
            operator=updated_list[max_index]
        )


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

