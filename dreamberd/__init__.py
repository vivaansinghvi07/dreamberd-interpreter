from time import sleep

from dreamberd.builtin import KEYWORDS
from dreamberd.processor.syntax_tree import CodeStatement
from dreamberd.interpreter import interpret_code_statements, load_globals

def main(filename: str, code: str, statements: list[tuple[CodeStatement, ...]]) -> None:  # idk what else to call this
    load_globals(filename, code, {}, set())
    namespace = KEYWORDS.copy()
    interpret_code_statements(statements, [namespace], [])
    print("\033[33mCode has finished executing. Press ^C once or twice to stop waiting for when-statements and after-statements.\033[039m")
    try:
        while True:
            sleep(1)  # just waiting for any clicks, when statements, etc
    except KeyboardInterrupt:
        exit()  # quit silently
