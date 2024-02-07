import sys
from argparse import ArgumentParser
from time import sleep
from typing import Union
from dreamberd.base import InterpretationError, Token, TokenType

from dreamberd.builtin import KEYWORDS, Name, Variable
from dreamberd.processor.lexer import tokenize
from dreamberd.processor.syntax_tree import CodeStatement, generate_syntax_tree
from dreamberd.interpreter import interpret_code_statements, load_global_dreamberd_variables, load_globals, load_public_global_variables

__all__ = ['run_repl', 'run_file']

__REPL_FILENAME = "__repl__"
sys.setrecursionlimit(100000)

def __get_next_repl_input(closed_scope_layers: int = 0) -> tuple[str, list[Token]]:
    print("   " * closed_scope_layers, '\033[33m>\033[39m ', end="", sep="")
    code = input()
    tokens = tokenize(__REPL_FILENAME, code)
    start_closed_scope_layers = closed_scope_layers
    for t in tokens:
        if t.type == TokenType.L_CURLY:
            closed_scope_layers += 1
        elif t.type == TokenType.R_CURLY:
            closed_scope_layers -= 1
    if closed_scope_layers < 0:
        raise InterpretationError("Too many closed braces with not enough open braces.")
    kickback_layers, i = 0, 0
    trimmed_code = code.strip()
    while i < len(trimmed_code) and trimmed_code[i] == '}':
        kickback_layers += 1
        i += 1
    if kickback_layers:
        print("\033[1A\r", "   " * (start_closed_scope_layers - kickback_layers), '\033[33m>\033[39m ', code, "   " * (start_closed_scope_layers), sep="", flush=True)
    if closed_scope_layers:
        new_code, new_tokens = __get_next_repl_input(closed_scope_layers) 
        code += new_code 
        tokens += new_tokens
    match l := [t for t in tokens if not t.type == TokenType.WHITESPACE]:
        case [Token(type=TokenType.NAME)]:
            tokens = [Token(TokenType.NAME, "print", -1, -1), Token(TokenType.WHITESPACE, " ", -1, -1), l[0], Token(TokenType.BANG, '!', -1, -1)]
    return code, tokens

def run_repl() -> None:
    namespaces: list[dict[str, Union[Variable, Name]]] = [KEYWORDS.copy()]  # type: ignore
    load_globals(__REPL_FILENAME, "", {}, set())
    load_global_dreamberd_variables(namespaces)
    async_statements = []
    when_statement_watchers = [{}]
    while True: 
        try:
            code, tokens = __get_next_repl_input()
            statements = generate_syntax_tree(__REPL_FILENAME, tokens, code)
            interpret_code_statements(statements, namespaces, async_statements, when_statement_watchers)
        except InterpretationError as e:
            print(e)
            
def run_file(filename: str) -> None:  # idk what else to call this

    with open(filename, 'r') as f:
        code = f.read()
    tokens = tokenize(filename, code)
    statements = generate_syntax_tree(filename, tokens, code)

    namespaces: list[dict[str, Union[Variable, Name]]] = [KEYWORDS.copy()]   # type: ignore
    load_globals(filename, code, {}, set())
    load_global_dreamberd_variables(namespaces)
    load_public_global_variables(namespaces)
    interpret_code_statements(statements, namespaces, [], [{}])
    print("\033[33mCode has finished executing. Press ^C once or twice to stop waiting for when-statements and after-statements.\033[039m")
    try:
        while True:
            sleep(1)  # just waiting for any clicks, when statements, etc
    except KeyboardInterrupt:
        exit()  # quit silently

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('file', help="The file containing your DreamBerd code.", nargs='?', default='', type=str)
    parser.add_argument('-s', '--show-traceback', help="Limit the error traceback to a single message.", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    if not args.show_traceback:
        sys.tracebacklimit = 0
    if not args.file:
        run_repl()
    else:
        run_file(args.file)

if __name__ == "__main__":
    main()
