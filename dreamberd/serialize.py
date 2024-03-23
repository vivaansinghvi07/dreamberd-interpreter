import json
import dataclasses
from typing import Any, Callable, Union, assert_never
from dreamberd.base import NonFormattedError, Token, TokenType

from dreamberd.builtin import KEYWORDS, BuiltinFunction, DreamberdFunction, DreamberdList, DreamberdNumber, DreamberdString, Name, Value, Variable
from dreamberd.interpreter import interpret_code_statements, load_globals
from dreamberd.processor.lexer import tokenize
from dreamberd.processor.syntax_tree import CodeStatement, VariableAssignment, generate_syntax_tree

SerializedDict = dict[str, Union[str, dict, list]]
DataclassSerializations = Union[Name, Variable, Value, CodeStatement, Token]

def serialize_obj(obj: Any) -> SerializedDict:
    match obj:
        case Name() | Variable() | Value() | CodeStatement() | Token(): return serialize_dreamberd_obj(obj)
        case _: return serialize_python_obj(obj)

def serialize_python_obj(obj: Any) ->  dict[str, Union[str, dict, list]]:
    match obj:
        case TokenType(): val = obj.value
        case dict(): 
            if not all(isinstance(k, str) for k in obj):
                raise NonFormattedError("Serialization Error: Encountered non-string dictionary keys.")
            val = {k: serialize_obj(v) for k, v in obj.items()}
        case list() | tuple(): val = [serialize_obj(x) for x in obj]
        case str(): val = obj
        case int() | float(): val = str(obj)
        case func if isinstance(func, Callable): val = func.__name__
        case _: assert_never(obj)
    return {
        "python_obj_type": type(obj).__name__,
        "value": val
    }

def serialize_dreamberd_obj(val: DataclassSerializations) -> dict[str, Union[str, dict, list]]:
    return {
        "dreamberd_obj_type": type(val).__name__,
        "attributes": [
            {
                "name": field.name,
                "value": serialize_obj(getattr(val, field.name))
            }
            for field in dataclasses.fields(val)  # type: ignore
        ]
    }

if __name__ == "__main__":

    code = """f main() => {
      print "Hello world"!
      }"""
    tokens = tokenize("", code)
    statements = generate_syntax_tree("", tokens, code)
    func_ns = KEYWORDS.copy()
    load_globals("", "", {}, set(), [], {})
    interpret_code_statements(statements, [func_ns], [], [{}])

    list_test_case = DreamberdList([
        DreamberdString("Hello world!"),
        DreamberdNumber(123.45), 
        func_ns["main"].value
    ])
    __import__('pprint').pprint(serialize_obj(list_test_case))
        
