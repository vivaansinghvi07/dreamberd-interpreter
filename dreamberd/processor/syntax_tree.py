from abc import ABCMeta
from typing import Optional
from dataclasses import dataclass

from dreamberd.base import STR_TO_OPERATOR, Token, TokenType, raise_error_at_line, raise_error_at_token

class CodeStatement(metaclass=ABCMeta):
    pass

# name name argname, argname, argname => {  ...
# for single line arrows, this is:
# name name argname, => expression !?
@dataclass
class FunctionDefinition(CodeStatement):
    keywords: list[str]
    name: str
    args: list[str]
    code: list[tuple[CodeStatement, ...]]
    is_async: bool

# name name {
@dataclass
class ClassDeclaration(CodeStatement):
    keyword: str
    name: str 
    code: list[tuple[CodeStatement, ...]]

# : ... can be ignored
# name name name: ... = something!?
@dataclass
class VariableAssignment(CodeStatement):
    name: str
    modifiers: list[str]
    lifetime: Optional[str]
    expression: list[Token]
    debug: bool
    confidence: int  ## amount of !!! after the decl for priority

# name expression { 
# since an expression can be a name:
#   name name {  
# can be both, and would have to be determined at runtime
@dataclass
class Conditional(CodeStatement):
    keyword: str
    condition: list[Token]
    code: list[tuple[CodeStatement, ...]]

# name expression !?
@dataclass
class ReturnStatement(CodeStatement):
    keyword: str
    expression: list[Token]
    debug: bool

# expression !?   < virtually indistinguishable from a return statement from a parsing perspective
@dataclass
class ExpressionStatement(CodeStatement):
    expression: list[Token]
    debug: bool

# name name = expression { 
@dataclass
class WhenStatement(CodeStatement):
    keyword: str
    name: str 
    expression: list[Token]
    code: list[tuple[CodeStatement, ...]]

# idea: create a class that evaluates at runtime what a statement is, so then execute it 
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

        # this is the start of a new scope, we don't care about those for rn in terms of starting a new statement
        if token.type == TokenType.L_CURLY:
            bracket_layers += 1 
        elif token.type == TokenType.R_CURLY:
            bracket_layers -= 1 

        if token.type in [TokenType.R_CURLY, TokenType.BANG, TokenType.QUESTION] and bracket_layers == 0:
            while statements[-1][-1].type == TokenType.NEWLINE:  # remove newlines at the end of a statement
                statements[-1].pop()
            statements.append([])
    
    # remove stray newlines cause they are annoying and shit, also remove empty tings
    final_statements = []
    for statement in statements:
        while statement and statement[-1].type in {TokenType.WHITESPACE, TokenType.NEWLINE}:  ## NOTE: END WILL NEVER BE WHITESPACE
            statement.pop()
        while statement and statement[0].type in {TokenType.WHITESPACE, TokenType.NEWLINE}:  ## NOTE: NOW START WILL NEVER BE WHITESPACE EITHER
            statement.pop(0)
        if not statement:
            continue
        final_statements.append(statement)

    return final_statements

def remove_type_hints(filename: str, code: str, statements: list[list[Token]]) -> list[list[Token]]:
    new_statements = []
    for tokens in statements:
        new_tokens = []
        adding_tokens = True 
        scope_layers, square_bracket_layers = 0, 0
        ref_square_bracket_layers = 0 
        curr = 0

        while curr < len(tokens):
            t = tokens[curr]

            # handle brackets
            if t.type == TokenType.L_CURLY: scope_layers += 1
            elif t.type == TokenType.R_CURLY: scope_layers -= 1
            if t.type == TokenType.L_SQUARE: square_bracket_layers += 1
            elif t.type == TokenType.R_SQUARE: square_bracket_layers -= 1


            # must be in the right place to consider removal
            if t.type == TokenType.COLON and scope_layers == 0:
                adding_tokens = False
                ref_square_bracket_layers = square_bracket_layers  # prob gonna be zero but idek imma just do this
            if not adding_tokens:

                # check if it is at an operator 
                if STR_TO_OPERATOR.get(t.value) and square_bracket_layers == ref_square_bracket_layers:
                    adding_tokens = True

                # adjust for Name<...> things, which also allows regex to pass too 
                elif t.type == TokenType.NAME and curr + 1 < len(tokens) and tokens[curr + 1].type == TokenType.LESS_THAN:
                    try:
                        while tokens[curr + 1].type != TokenType.GREATER_THAN:
                            curr += 1
                        curr += 1
                    except IndexError:
                        raise_error_at_token(filename, code, "Something went wrong parsing type hints (a.k.a. removing them).", t)

            if adding_tokens:  # cannot be elif because adding_tokens can be modified in the previous statement
                new_tokens.append(t)

            curr += 1
        
        new_statements.append(new_tokens)
    
    return new_statements

def assert_proper_indentation(filename: str, tokens: list[Token], code: str) -> None:
    looking_for_whitespace = False
    for t in tokens:
        if not looking_for_whitespace:
            if t.type == TokenType.NEWLINE:
                looking_for_whitespace = True
        else:
            if t.type == TokenType.WHITESPACE and len(t.value.replace('\t', '  ')) % 3:
                raise_error_at_token(filename, code, "Invalid indenting detected (must be a multiple of 3). Tabs count as 2 spaces.", t)
            looking_for_whitespace = False

def create_function_definition(filename: str, without_whitespace: list[Token], code: str, statements_inside_scope: list[tuple[CodeStatement, ...]]) -> tuple[CodeStatement, ...]:

    # parse this more to make sure it really really really can be a function
    names_in_row = [], other_names = []
    looking_for_in_row = True
    for t in without_whitespace:
        if t.type == TokenType.FUNC_POINT:
            break
        if looking_for_in_row:
            if t.type != TokenType.NAME:
                looking_for_in_row = False
            names_in_row.append(t)
        else:
            if t.type == TokenType.NAME:
                other_names.append(t)
            elif t.type != TokenType.COMMA:
                raise_error_at_token(filename, code, "Invalid token in function declaration.", t)

    # too many or too little keywords for the function
    if not 2 <= len(names_in_row) <= 4:
        raise_error_at_token(filename, code, "Insufficient keyword count in function declaration.", without_whitespace[0])

    is_async = len(names_in_row) == 4 
    can_be_async =  len(names_in_row) == 3 and not other_names
    if not is_async and can_be_async: # can be one of two forms: 

        return FunctionDefinition(      # func name(arg))
            keywords = names_in_row[:1],  
            name = names_in_row[1],
            args = names_in_row[2:],
            code = statements_inside_scope,
            is_async = False
        ), FunctionDefinition(           # async func name()
            keywords = names_in_row[:2],
            name = names_in_row[2],
            args = [],
            code = statements_inside_scope,
            is_async = True
        )

    elif is_async:
        
        # async func name (arg, arg, arg) ... 
        return FunctionDefinition(  
            keywords = names_in_row[:2],
            name = names_in_row[2],
            args = names_in_row[3:] + other_names,
            code = statements_inside_scope,
            is_async = True
        ),

    else:

        # func name (arg, arg, arg)
        return FunctionDefinition(
            keywords = names_in_row[:1],
            name = names_in_row[1],
            args = names_in_row[2:] + other_names,
            code = statements_inside_scope,
            is_async = False
        ),

def create_scoped_code_statement(filename: str, tokens: list[Token], without_whitespace: list[Token], code: str) -> tuple[CodeStatement, ...]:

    # this means that a scope is detected in the statement
    ends_with_punc = tokens[-1].type in {TokenType.BANG, TokenType.QUESTION}
    if tokens[-1 - int(ends_with_punc)].type != TokenType.R_CURLY:  # end will never be whitespace 
        raise_error_at_token(filename, code, "End of statement with open scope must close the scope.", tokens[-1])

    # at this point, this can be the "when" keyword, a class dec, a function call, or an if statement
    scope_open_index = [t.type == TokenType.L_CURLY for t in tokens].index(True)
    stuff_inside_scope = tokens[scope_open_index + 1 : len(tokens) - ends_with_punc - 1]
    statements_inside_scope = generate_syntax_tree(filename, stuff_inside_scope, code)
    
    # see the function pointer -> immediately know
    can_be_function = any([t.type == TokenType.FUNC_POINT for i, t in enumerate(without_whitespace) if i < scope_open_index])

    # now check for the shape of the when keyword
    can_be_when = without_whitespace[0].type == TokenType.NAME and \
                  without_whitespace[1].type == TokenType.NAME and \
                  without_whitespace[2].type in {
        TokenType.GREATER_THAN, TokenType.GREATER_EQUAL, 
        TokenType.EQUAL, TokenType.NOT_EQUAL, TokenType.LESS_THAN, TokenType.LESS_EQUAL
    }

    # now finally, check for classes
    can_be_class = without_whitespace[0].type == TokenType.NAME and \
                   without_whitespace[1].type == TokenType.NAME and \
                   without_whitespace[2].type == TokenType.R_CURLY

    # this dude is seperated to another function because the same code is reused in () => ... functions (no scope)
    if can_be_function:
        return create_function_definition(filename, without_whitespace, code, statements_inside_scope)

    elif can_be_when:
    
        # build thing for the when statement
        when_keywords, curr = [], 0
        while len(when_keywords) < 2:
            t = tokens[curr]
            if t.type != TokenType.WHITESPACE:
                when_keywords.append(t.value)
            curr += 1
        keyword, name = when_keywords
        expression = tokens[curr - 1 : scope_open_index]

        return WhenStatement(
            keyword = keyword,
            name = name, 
            expression = expression,
            code = statements_inside_scope
        ), Conditional(
            keyword = keyword,
            condition = expression,
            code = statements_inside_scope
        )

    elif can_be_class:

        # build thing for the class statement
        return ClassDeclaration(
            keyword = without_whitespace[0].value,
            name = without_whitespace[1].value,
            code = statements_inside_scope
        ), Conditional(
            keyword = without_whitespace[0].value,
            condition = without_whitespace[1:],
            code = statements_inside_scope
        )

    else:

        return Conditional(
            keyword = without_whitespace[0].value,
            condition = tokens[int(tokens[0].type == TokenType.WHITESPACE) + 1 : scope_open_index],
            code = statements_inside_scope
        ),

def create_unscoped_code_statement(filename: str, tokens: list[Token], without_whitespace: list[Token], code: str) -> tuple[CodeStatement, ...]: 
    
    is_debug = tokens[-1].type == TokenType.QUESTION
    confidence = 0 if is_debug else len(tokens[-1].value)

    # it's a function!!!!!!!!!!!!!!!!!
    if any(l := [t.type == TokenType.FUNC_POINT for t in tokens]):
        func_point_index = l.index(True)
        return create_function_definition(filename, without_whitespace, code, [(ReturnStatement(
            keyword = "",
            expression = tokens[func_point_index + 1 : -1],
            debug = is_debug
        ),)])
    
    # let's see what can be what D:
    can_be_return = without_whitespace[0].type == TokenType.NAME
    names_in_row = []
    lifetime = None
    looking_for_lifetime, can_be_var_declaration = False, any(l := [t.type == TokenType.EQUAL and t.value == '=' for t in tokens]) 
    for t in without_whitespace:
        if not can_be_var_declaration:
            break
        if not looking_for_lifetime:
            if t.type != TokenType.NAME:
                if t.type == TokenType.LESS_THAN:
                    looking_for_lifetime = True
                else:
                    break
            names_in_row.append(t)
        else:
            if t.type == TokenType.GREATER_THAN or not 3 <= len(names_in_row) <= 4 or not can_be_var_declaration:
                break
            if not lifetime:
                lifetime = t.value 
    
    can_be_var_declaration &= 3 <= len(names_in_row) <= 4 or len(names_in_row) == 1
    
    # make a list of all possible things, starting with plain expression 
    possibilities: list[CodeStatement] = [ExpressionStatement(tokens[:-1], is_debug)] 
    if can_be_return:
        possibilities.append(ReturnStatement(
            keyword = without_whitespace[0].value,   # should be the same as tokens[0].value but this makes me feel safe
            expression = tokens[1:],
            debug = is_debug
        ))
    if can_be_var_declaration:
        possibilities.append(VariableAssignment(
            name = names_in_row[-1],
            modifiers = names_in_row[:-1],
            lifetime = lifetime, 
            expression = tokens[l.index(True) + 1 : -1],   # the end should be a puncutation
            debug = is_debug, 
            confidence = confidence
        ))
    return tuple(possibilities)


def generate_syntax_tree(filename: str, tokens: list[Token], code: str) -> list[tuple[CodeStatement, ...]]: 
    """ Split the code up into lines, which are then parsed and shit """

    assert_proper_indentation(filename, tokens, code)
    statements = split_into_statements(tokens)
    removed_hints = remove_type_hints(filename, code, statements)
    final_statements = []
    
    # now we need to perform pattern matching on each list of statements
    for tokens in removed_hints:

        without_whitespace = [t for t in tokens if t.type != TokenType.WHITESPACE]

        try:
            # contains an open scope :)
            if any(t.type == TokenType.L_CURLY for t in tokens):
                final_statements.append(create_scoped_code_statement(
                    filename, tokens, without_whitespace, code
                ))

            else:  # options here are plain expression, variable assignment, return, or simple one-liner funciton expression
                final_statements.append(create_unscoped_code_statement(
                    filename, tokens, without_whitespace, code
                ))
            
            # exit if some possiblity was found 
            if final_statements[-1]:
                continue
        except IndexError:  # i have no idea what kind of errors are going to be rasied here
            pass
        raise_error_at_line(filename, code, without_whitespace[0].line, "Error parsing statement. I have no idea what went wrong, double check it and try again.")
        
    return final_statements
            
