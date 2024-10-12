from __future__ import annotations
import functools

import math
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional, Union
from dreamberd.base import NonFormattedError

from dreamberd.processor.syntax_tree import CodeStatement

__all__ = [
    'DreamberdFunction',
    'BuiltinFunction',
    'DreamberdList',
    'DreamberdNumber',
    'DreamberdString',
    'DreamberdBoolean',
    'DreamberdUndefined',
    'DreamberdSpecialBlankValue',
    'DreamberdObject',
    'DreamberdMap',
    'DreamberdKeyword',
    'DreamberdPromise'
]

FLOAT_TO_INT_PREC = 0.00000001
def is_int(x: Union[float, int]) -> bool:
    return min(x % 1, 1 - x % 1) < FLOAT_TO_INT_PREC

def db_not(x: DreamberdBoolean) -> DreamberdBoolean:
    if x.value is None:
        return DreamberdBoolean(None)
    return DreamberdBoolean(not x.value)

def db_list_push(self: DreamberdList, val: DreamberdValue) -> None:
    self.indexer[max(self.indexer.keys())+1] = len(self.values)-1
    self.values.append(val) 
    self.create_namespace()  # update the length

def db_list_pop(self: DreamberdList, index: Union[DreamberdNumber, DreamberdSpecialBlankValue]) -> DreamberdValue:
    if isinstance(index, DreamberdSpecialBlankValue):
        retval = self.values.pop()
        self.create_namespace()
        return retval
    elif not isinstance(index, DreamberdNumber) or not is_int(index.value):
        raise NonFormattedError("Expected integer for list popping.")
    elif not -1 <= index.value <= len(self.values) - 1:
        raise NonFormattedError("Indexing out of list bounds.")
    retval = self.values.pop(round(index.value) + 1)
    self.create_namespace()
    return retval

def db_str_push(self: DreamberdString, val: DreamberdValue) -> None:
    val_str = db_to_string(val).value
    max_user_index = max(self.indexer.keys())
    if len(val_str)>1:
        self.indexer[max_user_index+1] = (len(self.value)-1,val_str[1:])
    else:
        self.indexer[max_user_index+1] = (len(self.value)-1,"")
    self.value += val_str 
    #print(max(self.indexer.keys())+1)
    self.create_namespace()  # update the length

def db_str_pop(self: DreamberdString, index: Union[DreamberdNumber, DreamberdSpecialBlankValue]) -> DreamberdValue:
    if isinstance(index, DreamberdSpecialBlankValue):
        retval = self.value[-1]
        self.value = self.value[:-1]
        return DreamberdString(retval)
    elif not isinstance(index, DreamberdNumber) or not is_int(index.value):
        raise NonFormattedError("Expected integer for string popping.")
    elif not -1 <= index.value <= len(self.value) - 1:
        raise NonFormattedError("Indexing out of string bounds.")
    index_val = round(index.value) + 1
    retval = self.value[index_val]
    self.value = self.value[:index_val] + self.value[index_val + 1:]
    return DreamberdString(retval)

# class Value(metaclass=ABCMeta):   # TODO POTENTIALLY DO THIS TO ALLOW FOR MORE OBJECTS WITHOUT MUCH HASSLE
#     @abstractmethod 
#     def to_bool(self) -> Value: pass
#     @abstractmethod 
#     def to_num(self) -> Value: pass
#     @abstractmethod 
#     def to_str(self) -> Value: pass

class DreamberdValue():  # base class for shit  
    pass

class DreamberdMutable(DreamberdValue):  # mutable values
    pass

class DreamberdIndexable(DreamberdValue, metaclass=ABCMeta):
    
    @abstractmethod 
    def access_index(self, index: DreamberdValue) -> DreamberdValue: pass

    @abstractmethod
    def assign_index(self, index: DreamberdValue, val: DreamberdValue) -> None: pass

class DreamberdNamespaceable(DreamberdValue, metaclass=ABCMeta):
    namespace: dict[str, Union[Name, Variable]]

@dataclass 
class DreamberdFunction(DreamberdValue):  
    args: list[str]
    code: list[tuple[CodeStatement, ...]]
    is_async: bool

@dataclass
class BuiltinFunction(DreamberdValue):
    arg_count: int
    function: Callable
    modifies_caller: bool = False

@dataclass 
class DreamberdList(DreamberdIndexable, DreamberdNamespaceable, DreamberdMutable, DreamberdValue):
    values: list[DreamberdValue]
    indexer: dict[float, int] = field(init = False) # used for converting the user decimal indecies to the real indecies  
    namespace: dict[str, Union[Name, Variable]] = field(default_factory=dict)

    def __post_init__(self):
        self.create_namespace(False)
        self.indexer = dict()
        for index in range(-1, len(self.values)-1):
            self.indexer[index] = index

    def create_namespace(self, is_update: bool = True) -> None:

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

    def access_index(self, index: DreamberdValue) -> DreamberdValue:
        if not isinstance(index, DreamberdNumber):
            raise NonFormattedError("Cannot index a list with a non-number value.")
        if not -1 <= index.value <= len(self.values) - 1:
                raise NonFormattedError("Indexing out of list bounds.")
        elif index.value not in self.indexer:
            raise NonFormattedError("No value assigned to that index") # if inbounds index doesnt have assigned val
        user_index = index.value
        #print("user index:" + str(user_index))
        realIndex = self.indexer.get(user_index)
        #print("real index:" + str(realIndex))
        return self.values[round(realIndex) + 1]

    def assign_index(self, index: DreamberdValue, val: DreamberdValue) -> None:
        if not isinstance(index, DreamberdNumber):
            raise NonFormattedError("Cannot index a list with a non-number value.")
        if index.value in self.indexer:
            if not -1 <= index.value <= len(self.values) - 1:
                raise NonFormattedError("Indexing out of list bounds.")
            self.values[round(index.value) + 1] = val
            self.indexer[round(index.value) + 1] = round(index.value) + 1 # if adding to end, user index is real index
        else:  # assign in the middle of the array
            if not -1 <= index.value <= len(self.values) - 1:
                raise NonFormattedError("Indexing out of list bounds.")
            nearest_int_down = round(max((index.value + 2) // 1, 0))
            self.values[nearest_int_down:nearest_int_down] = [val]
            self.indexer[index.value] = nearest_int_down - 1 # if adding to end, user index is real index
            self.create_namespace()
            # all real indexes after the inserted item need 1 to be added to them
            user_indicies = self.indexer.keys()
            for user_index in user_indicies:
                if user_index > index.value:
                    self.indexer[user_index] += 1

@dataclass(unsafe_hash=True)
class DreamberdNumber(DreamberdIndexable, DreamberdMutable, DreamberdValue):
    value: Union[int, float]

    def _get_self_str(self) -> str:
        return str(self.value).replace('.', '').replace('-', '')

    def access_index(self, index: DreamberdValue) -> DreamberdValue:
        self_val_str = self._get_self_str()
        if not isinstance(index, DreamberdNumber):
            raise NonFormattedError("Cannot index a number with a non-number value.")
        if not is_int(index.value):
            raise NonFormattedError("Expected integer for number indexing.")
        elif not -1 <= index.value <= len(self_val_str) - 1:
            raise NonFormattedError("Indexing out of number bounds.")
        return DreamberdNumber(int(self_val_str[round(index.value) + 1]))

    def assign_index(self, index: DreamberdValue, val: DreamberdValue) -> None:
        self_val_str = self._get_self_str()
        sign = self.value / abs(self.value)
        if not is_int(self.value):
            raise NonFormattedError("Cannot assign into a non-interger number.")
        if not isinstance(index, DreamberdNumber):
            raise NonFormattedError("Cannot index a number with a non-number value.")
        if not isinstance(val, DreamberdNumber) or not is_int(val.value) or not 0 <= val.value <= 9:
            raise NonFormattedError("Cannot assign into a number with a non-integer value.")
        if is_int(index.value):
            if not -1 <= index.value <= len(self_val_str) - 1:
                raise NonFormattedError("Indexing out of number bounds.")
            index_num = round(index.value) + 1
            self.value = sign * int(self_val_str[:index_num] + str(round(val.value)) + self_val_str[index_num + 1:])
        else:  # assign in the middle of the array
            index_num = round(max((index.value + 2) // 1, 0))
            self.value = sign * int(self_val_str[:index_num] + str(round(val.value)) + self_val_str[index_num:])

@dataclass(unsafe_hash=True)
class DreamberdString(DreamberdIndexable, DreamberdNamespaceable, DreamberdMutable, DreamberdValue):
    value: str = field(hash=True)
    indexer: dict[float,tuple] = field(init = False,hash=False) # used for converting the user decimal indecies to the real indecies  
                                                                # tuple stores the real index in the first slot and any extra characters in the second
    namespace: dict[str, Union[Name, Variable]] = field(default_factory=dict, hash=False)

    def __post_init__(self):
        self.create_namespace(False)
        self.indexer = dict()
        for index in range(len(self.value)):
            self.indexer[index-1] = (index-1,"")

    def create_namespace(self, is_update: bool = True):
        if not is_update:
            self.namespace |= {
                'push': Name('push', BuiltinFunction(2, db_str_push, True)),
                'length': Name('length', DreamberdNumber(len(self.value))),
            }
        else:
            self.namespace['length'] = Name('length', DreamberdNumber(len(self.value)))

    def access_index(self, index: DreamberdValue) -> DreamberdValue:
        if not isinstance(index, DreamberdNumber):
            raise NonFormattedError("Cannot index a string with a non-number value.")
        #if not is_int(index.value):
        #    raise NonFormattedError("Expected integer for string indexing.")
        if not -1 <= index.value <= len(self.value) - 1:
            raise NonFormattedError("Indexing out of string bounds.")
        elif index.value not in self.indexer:
            raise NonFormattedError("No value assigned to that index") # if inbounds index doesnt have assigned val
        user_index = index.value
        index_data = self.indexer[user_index]
        real_index = index_data[0]
        extra_characters = index_data[1]
        return self.value[real_index+1]+extra_characters

    def assign_index(self, index: DreamberdValue, val: DreamberdValue) -> None:
        if not isinstance(index, DreamberdNumber):
            raise NonFormattedError("Cannot index a string with a non-number value.")
        val_str = db_to_string(val).value
        if index.value in self.indexer:
            ## add, when modifying, reduce user indexes by the length of the replaced index's extra characters
            indexer_data = self.indexer[index.value]
            index_num = indexer_data[0]+1
            excess_length = len(indexer_data[1])
            self.value = self.value[:index_num] + val_str + self.value[index_num + excess_length + 1:]
            if len(val_str)>1:
                indexer_data = (indexer_data[0],val_str[:-1])
            else:
                indexer_data = (indexer_data[0],"")
            self.indexer[index.value] = indexer_data
            user_indicies = self.indexer.keys()
            for user_index in user_indicies:
                if user_index > index.value:
                    indexer_data = self.indexer[user_index]
                    indexer_data = (indexer_data[0]-excess_length,indexer_data[1])
                    self.indexer[user_index] = indexer_data
            self.create_namespace();

        else:  # assign in the middle of the array
            if not -1 <= index.value <= len(self.value) - 1:
                raise NonFormattedError("Indexing out of string bounds.")
            index_num = round(max((index.value + 2) // 1, 0))
            self.value = self.value[:index_num] + val_str + self.value[index_num:]
            if len(val_str)>1:
                indexer_data = (index_num-1,val_str[1:])
            else:
                indexer_data = (index_num -1, "")
            self.indexer[index.value] = indexer_data
            user_indicies = self.indexer.keys()
            for user_index in user_indicies:
                if user_index > index.value:
                    #print(f"updating user index {user_index},{self.indexer[user_index]}")
                    indexer_data = self.indexer[user_index]
                    indexer_data = (indexer_data[0]+len(val_str),indexer_data[1])
                    self.indexer[user_index] = indexer_data
            self.create_namespace()

@dataclass 
class DreamberdBoolean(DreamberdValue):
    value: Optional[bool]  # none represents maybe?

@dataclass 
class DreamberdUndefined(DreamberdValue):
    pass
@dataclass 
class DreamberdSpecialBlankValue(DreamberdValue):
    pass

@dataclass 
class DreamberdObject(DreamberdNamespaceable, DreamberdValue):
    class_name: str
    namespace: dict[str, Union[Name, Variable]] = field(default_factory=dict)

@dataclass 
class DreamberdMap(DreamberdIndexable, DreamberdValue):
    self_dict: dict[Union[int, float, str], DreamberdValue]

    def access_index(self, index: DreamberdValue) -> DreamberdValue:
        if not isinstance(index, (DreamberdString, DreamberdNumber)):
            raise NonFormattedError("Keys of a map must be an index or a number.")
        return self.self_dict[index.value]

    def assign_index(self, index: DreamberdValue, val: DreamberdValue) -> None:
        if not isinstance(index, (DreamberdString, DreamberdNumber)):
            raise NonFormattedError("Keys of a map must be an index or a number.")
        self.self_dict[index.value] = val

@dataclass 
class DreamberdKeyword(DreamberdValue):
    value: str

@dataclass 
class DreamberdPromise(DreamberdValue):
    value: Optional[DreamberdValue]

@dataclass
class Name:
    name: str
    value: DreamberdValue

@dataclass 
class VariableLifetime:
    value: DreamberdValue
    lines_left: int 
    confidence: int
    can_be_reset: bool
    can_edit_value: bool

@dataclass
class Variable:
    name: str 
    lifetimes: list[VariableLifetime]
    prev_values: list[DreamberdValue]

    @property 
    def can_be_reset(self) -> bool:
        if self.lifetimes:
            return self.lifetimes[0].can_be_reset
        raise NonFormattedError("Variable is undefined.")

    @property 
    def can_edit_value(self) -> bool:
        if self.lifetimes:
            return self.lifetimes[0].can_edit_value
        raise NonFormattedError("Variable is undefined.")

    def add_lifetime(self, value: DreamberdValue, confidence: int, duration: int, can_be_reset: bool, can_edit_value: bool) -> None:
        for i in range(len(self.lifetimes) + 1):
            if i == len(self.lifetimes) or self.lifetimes[i].confidence == confidence:
                if i == 0:
                    self.prev_values.append(self.value)
                self.lifetimes[i:i] = [VariableLifetime(value, duration, confidence, can_be_reset, can_edit_value)]
                break

    def clear_outdated_lifetimes(self) -> None:
        remove_indeces = []
        for i, l in enumerate(self.lifetimes):
            if l.lines_left == 0:
                remove_indeces.append(i)
        for i in reversed(remove_indeces):
            del self.lifetimes[i]

    @property
    def value(self) -> DreamberdValue:
        if self.lifetimes:
            return self.lifetimes[0].value
        raise NonFormattedError("Variable is undefined.")
    
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
                                for n_ in range(2):
                                    keywords.add("".join([c * i for c, i in zip('function', [f,u,n,c,t,i,o,n_])]) or 'fn')    # the `or` allows the empty string to not count
    return list(keywords)

FUNCTION_KEYWORDS = all_function_keywords()
KEYWORDS = {kw: Name(kw, DreamberdKeyword(kw)) for kw in 
    ['class', 'className', 'after', 'const', 'var', 'when', 'if', 'async', 
     'return', 'delete', 'await', 'previous', 'next', 'reverse', 'export', 'import'] + FUNCTION_KEYWORDS}

############################################
##           DREAMBERD BUILTINS           ##
############################################

# this is for functions that return the same value, like current or new
def db_identity(val: DreamberdValue) -> DreamberdValue:
    return val

def db_map() -> DreamberdMap:
    return DreamberdMap({})

def db_to_boolean(val: DreamberdValue) -> DreamberdBoolean:
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

def db_to_string(val: DreamberdValue) -> DreamberdString:
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
        case DreamberdMap():
            return_string = f'{{{", ".join([f"{k}: {db_to_string(v).value}" for k, v in val.self_dict.items()])}}}'
    return DreamberdString(return_string)

def db_print(*vals: DreamberdValue) -> None:
    print(*[db_to_string(v).value for v in vals])

def db_to_number(val: DreamberdValue) -> DreamberdNumber:
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
                raise NonFormattedError("Cannot turn a non-empty list into a number.")
            return_number = 0 
        case DreamberdMap():
            if val.self_dict:
                raise NonFormattedError("Cannot turn a non-empty map into a number.")
            return_number = 0 
        case _:
            raise NonFormattedError(f"Cannot turn type {type(val).__name__} into a number.")
    return DreamberdNumber(return_number)

def db_signal(starting_value: DreamberdValue) -> DreamberdValue:
    obj = Name('', starting_value)
    def signal_func(setter_val: DreamberdValue) -> Optional[DreamberdValue]:
        nonlocal obj
        if isinstance(setter_val, DreamberdSpecialBlankValue):
            return obj.value
        obj.value = setter_val
    return BuiltinFunction(1, signal_func)

def db_exit() -> None:
    exit()

def __math_function_decorator(func: Callable):
    @functools.wraps(func)
    def inner(*args) -> DreamberdNumber:  # no kwargs
        for arg in args:
            if not isinstance(arg, DreamberdNumber):
                raise NonFormattedError("Cannot pass in a non-number value into a math function.")
        return DreamberdNumber(func(*[arg.value for arg in args]))
    return inner

def __number_function_maker(num: int) -> BuiltinFunction:
    def the_func(n: DreamberdNumber) -> DreamberdNumber:
        nonlocal num
        if not isinstance(n, DreamberdNumber):
            raise NonFormattedError(f"Expected a number in the ones digit. Instead received a {type(n).__name__}")
        return DreamberdNumber(num + n.value)
    return BuiltinFunction(1, the_func)

# get ready, this is boutta be crazy
MATH_FUNCTION_KEYWORDS = {
    name: Name(name, BuiltinFunction((-1 if any([arg[0] == '*' and len(arg) > 1 for arg in v.__text_signature__[1:-1].split(', ')]) 
                                         else len([arg for arg in v.__text_signature__[1:-1].split(', ') if arg.isalpha()])) 
        if v.__text_signature__ else 1 if name == 'log' else -1, __math_function_decorator(v)) 
    if isinstance(v := getattr(math, name), type(math.ulp)) else DreamberdNumber(v)) for name in dir(math) if not name.startswith('__')
}  # the frick is this
BUILTIN_FUNCTION_KEYWORDS = {
    "new": Name("new", BuiltinFunction(1, db_identity)),
    "current": Name("current", BuiltinFunction(1, db_identity)),
    "Map": Name("Map", BuiltinFunction(0, db_map)),
    "Boolean": Name("Boolean", BuiltinFunction(1, db_to_boolean)),
    "String": Name("String", BuiltinFunction(1, db_to_string)),
    "print": Name("print", BuiltinFunction(-1, db_print)),
    "exit": Name("exit", BuiltinFunction(0, db_exit)),
    "Number": Name("Number", BuiltinFunction(1, db_to_number)),
    "use": Name("use", BuiltinFunction(1, db_signal))
}
BUILTIN_VALUE_KEYWORDS = {
    "true": Name("true", DreamberdBoolean(True)),
    "maybe": Name("maybe", DreamberdBoolean(None)),
    "false": Name("false", DreamberdBoolean(False)),
    "undefined": Name("undefined", DreamberdUndefined()),
    "": Name("", DreamberdSpecialBlankValue())
}
NUMBER_NAME_KEYWORDS = {
    name: Name(name, DreamberdNumber(num)) for num, name in enumerate(["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
     "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "ninteen"])
} | {
    name: Name(name, __number_function_maker(num)) for num, name in zip(range(20, 100, 10), ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"])
}  # this is so cursed

KEYWORDS |= BUILTIN_FUNCTION_KEYWORDS | BUILTIN_VALUE_KEYWORDS | MATH_FUNCTION_KEYWORDS |\
            NUMBER_NAME_KEYWORDS
