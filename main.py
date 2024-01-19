#!/usr/bin/python3

import sys

import dreamberd.processor.expression_tree as db_exp_tree
import dreamberd.processor.lexer as db_lexer
import dreamberd.processor.syntax_tree as db_syn_tree

# sys.tracebacklimit = 0  # uncomment this line for db-specific error messages only

def test_expression():
    code = "await next name"
    tokens = db_lexer.tokenize("test.db", code)
    exp = db_exp_tree.build_expression_tree("test.db", tokens, code)
    print("\n ", code, "\n")
    print(exp.to_string(), "\n")

def main():
    with open(sys.argv[1]) as f:
        code = f.read()
    tokens = db_lexer.tokenize(sys.argv[1], code)[:-1]
    tree = db_syn_tree.generate_syntax_tree(sys.argv[1], tokens, code)
    print(*tree, sep="\n")

if __name__ == "__main__":
    main()
