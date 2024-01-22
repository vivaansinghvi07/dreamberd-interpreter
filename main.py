#!/usr/bin/env python3

import sys

from pprint import pprint
import dreamberd.processor.expression_tree as db_exp_tree
import dreamberd.processor.lexer as db_lexer
import dreamberd.processor.syntax_tree as db_syn_tree
from dreamberd import main as db_main 

sys.tracebacklimit = 0  # uncomment this line for db-specific error messages only

def test_expression():
    code = "[ ]"
    tokens = db_lexer.tokenize("test.db", code)
    exp = db_exp_tree.build_expression_tree("test.db", tokens, code)
    print("\n ", code, "\n")
    print(exp.to_string(), "\n")

def main():
    filename = sys.argv[1]
    with open(filename) as f:
        code = f.read()
    tokens = db_lexer.tokenize(filename, code)[:-1]
    tree = db_syn_tree.generate_syntax_tree(filename, tokens, code)
    # pprint(tree)
    db_main(filename, code, tree)

if __name__ == "__main__":
    main()
