#!/usr/bin/python3

import sys 

import lexer
from syntax_tree import build_expression_tree

# sys.tracebacklimit = 0

def main():
    pass

if __name__ == "__main__":
    code = "c  ^  func [1, 2, 3][[  4,  func 5,  other 6, 7  ][1]] ^ 7"
    tokens = lexer.tokenize("test.db", code)
    exp = build_expression_tree("test.db", tokens, code)
    print('\n ', code, '\n')
    print(exp.to_string(), '\n')
    # with open(sys.argv[1]) as f:
    #     code = f.read()
    # lexer.tokenize(sys.argv[1], code)
    # main()
