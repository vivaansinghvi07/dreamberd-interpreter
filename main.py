#!/usr/bin/python3

import sys 

import lexer
from syntax_tree import build_expression_tree

# sys.tracebacklimit = 0

def main():
    pass

if __name__ == "__main__":
    tokens = lexer.tokenize("test.db", "func 1 , [3,4,5] , 3^7  +  90")
    exp = build_expression_tree("test.db", tokens)
    print(exp.to_string())
    # with open(sys.argv[1]) as f:
    #     code = f.read()
    # lexer.tokenize(sys.argv[1], code)
    # main()
