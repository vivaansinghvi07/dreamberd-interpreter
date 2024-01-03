#!/usr/bin/python3

import sys 

import dreamberd.processor.lexer as db_lexer
import dreamberd.processor.expression_tree as db_exp_tree

# sys.tracebacklimit = 0

def main():
    pass

if __name__ == "__main__":
    code = "c  ^  -func [1, 2, 3][[  -4,  func 5,  other 6, 7  ][1]] ^ ;7"
    tokens = db_lexer.tokenize("test.db", code)
    exp = db_exp_tree.build_expression_tree("test.db", tokens, code)
    print('\n ', code, '\n')
    print(exp.to_string(), '\n')
    # with open(sys.argv[1]) as f:
    #     code = f.read()
    # lexer.tokenize(sys.argv[1], code)
    # main()
