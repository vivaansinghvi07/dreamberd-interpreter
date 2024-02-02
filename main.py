#!/usr/bin/env python3

from pprint import pprint
import dreamberd.processor.expression_tree as db_exp_tree
import dreamberd.processor.lexer as db_lexer
from dreamberd import main as db_main 

def test_expression():
    code = "[ ]"
    tokens = db_lexer.tokenize("test.db", code)
    exp = db_exp_tree.build_expression_tree("test.db", tokens, code)
    print("\n ", code, "\n")
    print(exp.to_string(), "\n")

if __name__ == "__main__":
    db_main()
