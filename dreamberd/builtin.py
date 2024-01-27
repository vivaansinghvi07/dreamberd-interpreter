from __future__ import annotations
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union
from dreamberd.base import InterpretationError

from dreamberd.processor.syntax_tree import CodeStatement

FLOAT_TO_INT_PREC = 0.00000001
def is_int(x: Union[float, int]) -> bool:
    return min(x % 1, 1 - x % 1) < FLOAT_TO_INT_PREC

def db_not(x: DreamberdBoolean) -> DreamberdBoolean:
    if x.value is None:
        return DreamberdBoolean(None)
    return DreamberdBoolean(not x.value)

# class Value(metaclass=ABCMeta):   # TODO POTENTIALLY DO THIS TO ALLOW FOR MORE OBJECTS WITHOUT MUCH HASSLE
#     @abstractmethod 
#     def to_bool(self) -> Value: pass
#     @abstractmethod 
#     def to_num(self) -> Value: pass
#     @abstractmethod 
#     def to_str(self) -> Value: pass

class Value():  # base class for shit  
    pass

class DreamberdMutable(Value):  # mutable values
    pass

class DreamberdIndexable(Value, metaclass=ABCMeta):
    
    @abstractmethod 
    def access_index(self, index: Value) -> Value: pass

    @abstractmethod
    def assign_index(self, index: Value, val: Value) -> None: pass

class DreamberdNamespaceable(Value, metaclass=ABCMeta):
    namespace: dict[str, Union[Name, Variable]]

@dataclass 
class DreamberdFunction(Value):  
    args: list[str]
    code: list[tuple[CodeStatement, ...]]
    is_async: bool

@dataclass
class BuiltinFunction(Value):
    arg_count: int
    function: Callable
    modifies_caller: bool = False

@dataclass 
class DreamberdList(DreamberdIndexable, DreamberdNamespaceable, DreamberdMutable):
    values: list[Value]
    namespace: dict[str, Union[Name, Variable]] = field(default_factory=dict)

    def __post_init__(self):
        self.create_namespace()

    def create_namespace(self, is_update: bool = False) -> None:

        def db_list_push(self, val: Value) -> None:
            self.values.append(val) 
            self.create_namespace(True)  # update the length

        def db_list_pop(self, index: DreamberdNumber) -> Value:
            if not is_int(index.value):
                raise InterpretationError("Expected integer for list popping.")
            elif not -1 <= index.value <= len(self.values) - 1:
                raise InterpretationError("Indexing out of list bounds.")
            self.namespace["length"] = Name('length', DreamberdNumber(len(self.values) - 1))
            return self.values.pop(round(index.value) - 1)

        if not is_update:
            self.namespace = {
                'push': Name('push', BuiltinFunction(2, db_list_push, True)),
                'pop': Name('pop', BuiltinFunction(2, db_list_pop, True)),
                'length': Name('length', DreamberdNumber(len(self.values))),
            }
        elif is_update:
            self.namespace |= {
                'length': Name('length', DreamberdNumber(len(self.values))),
            }

    def access_index(self, index: Value) -> Value:
        if not isinstance(index, DreamberdNumber):
            raise InterpretationError("Cannot index a list with a non-number value.")
        if not is_int(index.value):
            raise InterpretationError("Expected integer for list indexing.")
        elif not -1 <= index.value <= len(self.values) - 1:
            raise InterpretationError("Indexing out of list bounds.")
        return self.values[round(index.value) - 1]

    def assign_index(self, index: Value, val: Value) -> None:
        if not isinstance(index, DreamberdNumber):
            raise InterpretationError("Cannot index a list with a non-number value.")
        if is_int(index.value):
            if not -1 <= index.value <= len(self.values) - 1:
                raise InterpretationError("Indexing out of list bounds.")
            self.values[round(index.value) - 1] = val
        else:  # assign in the middle of the array
            nearest_int_down = max((index.value - 1) // 1, 0)
            self.values[nearest_int_down:nearest_int_down] = [val]
            self.create_namespace(True)

@dataclass(unsafe_hash=True)
class DreamberdNumber(DreamberdIndexable, DreamberdMutable):
    value: Union[int, float]

    def access_index(self, index: Value) -> Value:
        if not isinstance(index, DreamberdNumber):
            raise InterpretationError("Cannot index a number with a non-number value.")
        pass

    def assign_index(self, index: Value, val: Value) -> None:
        if not isinstance(index, DreamberdNumber):
            raise InterpretationError("Cannot index a number with a non-number value.")
        pass

@dataclass(unsafe_hash=True)
class DreamberdString(DreamberdIndexable, DreamberdNamespaceable):
    value: str = field(hash=True)
    namespace: dict[str, Union[Name, Variable]] = field(default_factory=dict, hash=False)

    def __post_init__(self):
        self.create_namespace()

    def create_namespace(self):
        self.namespace |= {
            'length': Name('length', DreamberdNumber(len(self.value))),
        }

    def access_index(self, index: Value) -> Value:
        pass

    def assign_index(self, index: Value, val: Value) -> None:
        pass

@dataclass 
class DreamberdBoolean(Value):
    value: Optional[bool]  # none represents maybe?

@dataclass 
class DreamberdUndefined(Value):
    pass

@dataclass 
class DreamberdObject(DreamberdNamespaceable):
    class_name: str
    namespace: dict[str, Union[Name, Variable]] = field(default_factory=dict)

@dataclass 
class DreamberdMap(DreamberdIndexable):
    self_dict: dict[Any, Any]

    def access_index(self, index: DreamberdNumber) -> Value:
        pass

    def assign_index(self, index: DreamberdNumber, val: Value) -> None:
        pass

@dataclass 
class DreamberdKeyword(Value):
    value: str

@dataclass 
class DreamberdPromise(Value):
    value: Optional[Value]

@dataclass
class Name:
    name: str
    value: Value

@dataclass 
class VariableLifetime:
    value: Value
    lines_left: int 
    confidence: int

@dataclass
class Variable:
    name: str 
    lifetimes: list[VariableLifetime]
    prev_values: list[Value]

    def add_lifetime(self, value: Value, confidence: int, duration: int) -> None:
        for i in range(len(self.lifetimes) + 1):
            if i == len(self.lifetimes) or self.lifetimes[i].confidence == confidence:
                if i == 0:
                    self.prev_values.append(self.value)
                self.lifetimes[i:i] = [VariableLifetime(value, duration, confidence)]
                break

    def clear_outdated_lifetimes(self) -> None:
        remove_indeces = []
        for i, l in enumerate(self.lifetimes):
            if l.lines_left == 0:
                remove_indeces.append(i)
        for i in reversed(remove_indeces):
            del self.lifetimes[i]

    @property
    def value(self) -> Value:
        if self.lifetimes:
            return self.lifetimes[0].value
        raise InterpretationError("Variable is undefined.")
    
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

FUNCTION_KEYWORDS = all_function_keywords()
KEYWORDS = {kw: Name(kw, DreamberdKeyword(kw)) for kw in ['class', 'className', 'const', 'var', 'when', 'if', 'async', 'return', 'delete', 'await', 'previous', 'next'] + FUNCTION_KEYWORDS}

############################################
##           DREAMBERD BUILTINS           ##
############################################

# the new function does absolutely nothing, lol
def db_new(val: Value) -> Value:
    return val

def db_map() -> DreamberdMap:
    return DreamberdMap({})

def db_to_boolean(val: Value) -> DreamberdBoolean:
    return_bool = None
    match val: 
        case DreamberdString():
            return_bool = bool(val.value.strip()) or (None if len(val.value) else False)
        case DreamberdNumber():  # maybe if it is 0.xxx, false if it is 0, true if anything else
            return_bool = bool(round(val.value)) or (None if abs(val.value) > FLOAT_TO_INT_PREC else False)
        case DreamberdList():
            return_bool = bool(val.values)
        case DreamberdMap():
            return_bool = bool(val.self_dict)
        case DreamberdBoolean():
            return_bool = val.value
        case DreamberdUndefined():
            return_bool = False
        case DreamberdFunction() | DreamberdObject() | DreamberdKeyword():
            return_bool = None  # maybe for these cause im mischevious
    return DreamberdBoolean(return_bool)

def db_to_string(val: Value) -> DreamberdString:
    return_string = str(val)
    match val:
        case DreamberdString():
            return_string = val.value
        case DreamberdList():
            return_string = f"[{', '.join([db_to_string(v).value for v in val.values])}]"
        case DreamberdBoolean():
            return_string = "true"  if val.value else \
                            "maybe" if val.value is None else "false"
        case DreamberdNumber():
            return_string = str(val.value)
        case DreamberdFunction(): 
            return_string = f"<function ({', '.join(val.args)})>"
        case DreamberdObject():
            return_string = f"<object {val.class_name}>" 
        case DreamberdUndefined():
            return_string = "undefined"
        case DreamberdKeyword():
            return_string = val.value
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
        case DreamberdUndefined():
            return_number = 0 
        case DreamberdBoolean():
            return_number = int(val.value is not None and val.value) + (val.value is None) * 0.5
        case DreamberdList():
            if val.values:
                raise InterpretationError("Cannot turn a non-empty list into a number.")
            return_number = 0 
        case DreamberdMap():
            if val.self_dict:
                raise InterpretationError("Cannot turn a non-empty map into a number.")
            return_number = 0 
        case _:
            raise InterpretationError(f"Cannot turn type {type(val).__name__} into a number.")
    return DreamberdNumber(return_number)

def db_exit() -> None:
    exit()

BUILTIN_FUNCTION_KEYWORDS = {
    "new": Name("new", BuiltinFunction(1, db_new)),
    "Map": Name("new", BuiltinFunction(0, db_map)),
    "Boolean": Name("Boolean", BuiltinFunction(1, db_to_boolean)),
    "String": Name("String", BuiltinFunction(1, db_to_string)),
    "print": Name("print", BuiltinFunction(-1, db_print)),
    "exit": Name("exit", BuiltinFunction(0, db_exit)),
    "Number": Name("Number", BuiltinFunction(1, db_to_number))
}
KEYWORDS |= BUILTIN_FUNCTION_KEYWORDS
