from __future__ import annotations
from abc import ABCMeta
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

from dreamberd.processor.syntax_tree import CodeStatement

class Value(metaclass=ABCMeta):
    pass

@dataclass 
class DreamberdFunction(Value):
    args: list[str]
    code: list[CodeStatement]

@dataclass
class BuiltinFunction(Value):
    arg_count: int
    function: Callable

@dataclass 
class DreamberdList(Value):
    values: list[Value]
    namespace: dict[str, Name] = field(default_factory=dict)

    def create_namespace(self, is_update: bool = False) -> None:

        def db_list_push(val: Value) -> None:
            self.values.append(val) 

        def db_list_pop(index: DreamberdNumber) -> Value:
            if not isinstance(index.value, int):
                raise TypeError("Expected integer for list popping.")
            elif not -1 <= index.value <= len(self.values) - 1:
                raise IndexError("Indexing out of list bounds.")
            return self.values.pop(index.value - 1)

        def db_list_assign(index: DreamberdNumber, val: Value) -> None:
            if isinstance(index.value, int):
                if not -1 <= index.value <= len(self.values) - 1:
                    raise IndexError("Indexing out of list bounds.")
                self.values[index.value - 1] = val
            else:  # assign in the middle of the array
                nearest_int_down = max(index.value // 1, 0)
                self.values[nearest_int_down:nearest_int_down] = [val]
    
        if not is_update:
            self.namespace = {
                'push': Name('push', BuiltinFunction(1, db_list_push)),
                'pop': Name('push', BuiltinFunction(1, db_list_pop)),
                'assign': Name('push', BuiltinFunction(1, db_list_assign)),
                'length': Name('length', DreamberdNumber(len(self.values))),
            }
        elif is_update:
            self.namespace |= {
                'length': Name('length', DreamberdNumber(len(self.values))),
            }

@dataclass 
class DreamberdNumber(Value):
    value: Union[int, float]

@dataclass 
class DreamberdString(Value):
    value: str 
    namespace: dict[str, Name] = field(default_factory=dict)

    def create_namespace(self, is_update: bool = False):
        def db_list_push(val: Value) -> None:
            self.value += db_to_string(val).value

        if not is_update:
            self.namespace = {
                'push': Name('push', BuiltinFunction(1, db_list_push)),
                'length': Name('length', DreamberdNumber(len(self.value))),
            }
        elif is_update:
            self.namespace |= {
                'length': Name('length', DreamberdNumber(len(self.value))),
            }

@dataclass 
class DreamberdBoolean(Value):
    value: Optional[bool]  # none represents maybe?

@dataclass 
class DreamberdObject(Value):
    class_name: str
    namespace: dict[str, Name] = field(default_factory=dict)

@dataclass 
class Keyword(Value):
    value: str

@dataclass
class Name:
    name: str
    value: Value

@dataclass 
class VariableLifetime:
    value: Value
    start: int
    duration: int 
    confidence: int

@dataclass
class Variable:
    name: str 
    lifetimes: list[VariableLifetime]

    def add_lifetime(self, value: Value, line: int, confidence: int, duration: int) -> None:
        for i in range(len(self.lifetimes)):
            if self.lifetimes[i].confidence == confidence:
                self.lifetimes[i:i] = [VariableLifetime(value, line, duration, confidence)]

    def clear_outdated_lifetimes(self, line: int) -> None:
        remove_indeces = []
        for i, l in enumerate(self.lifetimes):
            if line - l.start > l.duration:
                remove_indeces.append(i)
        for i in reversed(remove_indeces):
            del self.lifetimes[i]

    @property
    def value(self) -> Value:
        return self.lifetimes[0].value

def all_function_keywords() -> list[str]:

    # this code boutta be crazy
    # i refuse to use the builtin combinations
    keywords = set()
    for f in range(2):
        for u in range(2):
            for n in range(2):
                for c in range(2):
                    for t in range(2):
                        for i in range(2):
                            for o in range(2):
                                for n2 in range(2):
                                    keywords.add("".join([c * i for c, i in zip('function', [f, u, n, c, t, i, o, n2])]) or 'fn')
    return list(keywords)

function_keywords = all_function_keywords()
KEYWORDS = {kw: Name(kw, Keyword(kw)) for kw in ['class', 'className', 'const', 'var', 'when', 'if', 'async', 'return', 'delete'] + function_keywords}

############################################
##           DREAMBERD BUILTINS           ##
############################################

def db_to_string(val: Value) -> DreamberdString:
    return_string = str(val)
    match val:
        case DreamberdString():
            return_string = val.value
        case DreamberdList():
            return_string = str([db_to_string(v) for v in val.values])
        case DreamberdBoolean():
            return_string = "true"  if val.value else \
                            "maybe" if val.value is None else "false"
        case DreamberdNumber():
            return_string = str(val.value)
        case DreamberdFunction(): 
            return_string = f"<function ({', '.join(val.args)})>"
        case DreamberdObject():
            return_string = f"<object {val.class_name}>" 
    return DreamberdString(return_string)

def db_print(*vals: Value) -> None:
    print(*[db_to_string(v).value for v in vals])

def db_to_number(val: Value) -> DreamberdNumber:
    return_number = 0
    match val:
        case DreamberdNumber():
            return_number = val.value
        case DreamberdString():
            return_number = float(val.value)

def db_exit() -> None:
    exit()

def after() -> None:
    exit()
