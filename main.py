#!/usr/bin/python3

import sys 

import lexer
from syntax_tree import build_expression_tree

# sys.tracebacklimit = 0

def main():
    pass

if __name__ == "__main__":
    code = "1+1^3  *  func(a, b)"
    tokens = lexer.tokenize("test.db", code)
    exp = build_expression_tree("test.db", tokens)
    print('\n', code, '\n')
    print(exp.to_string(), '\n')
    # with open(sys.argv[1]) as f:
    #     code = f.read()
    # lexer.tokenize(sys.argv[1], code)
    # main()
