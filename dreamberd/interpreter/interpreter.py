# a note about this file: i've appended a "; raise" at the end of nearly every custom raise_error_at_token call, 
# because pyright (my nvim LSP) doesn't recognize the code terminating at the raise_error_at_token, so I add this 
# to make sure it recognizes that and doesn't yell at me because I don't like being yelled at

# TODO : consider turning name_watchers, filename, and code ino global variables

import re
import random
from time import sleep
from threading import Thread
from copy import copy, deepcopy
from difflib import SequenceMatcher
from typing import Optional, TypeAlias, Union

from dreamberd.base import InterpretationError, OperatorType, Token, raise_error_at_token
from dreamberd.interpreter.builtin import FLOAT_TO_INT_PREC, KEYWORDS, BuiltinFunction, DreamberdBoolean, DreamberdFunction, DreamberdKeyword, DreamberdList, DreamberdMap, DreamberdNumber, DreamberdObject, DreamberdPromise, DreamberdString, DreamberdUndefined, Name, Variable, Value, VariableLifetime, db_not, db_to_boolean, db_to_number, db_to_string, is_int
from dreamberd.processor.expression_tree import ExpressionTreeNode, FunctionNode, ListNode, ValueNode, IndexNode, ExpressionNode, build_expression_tree
from dreamberd.processor.syntax_tree import AfterStatement, ClassDeclaration, CodeStatement, Conditional, DeleteStatement, ExpressionStatement, FunctionDefinition, ReturnStatement, VariableAssignment, VariableDeclaration, WhenStatement

# several "ratios" used in the approx equal function
NUM_EQUALITY_RATIO = 0.1  # a-b / b 
STRING_EQUALITY_RATIO = 0.7  # min ratio to be considered equal
LIST_EQUALITY_RATIO = 0.7  # min ratio of all the elements of a list to be equal for the lists to be equal
MAP_EQUALITY_RATIO = 0.6  # lower thresh cause i feel like it
FUNCTION_EQUALITY_RATIO = 0.6  # yeah 
OBJECT_EQUALITY_RATIO = 0.6 

Namespace: TypeAlias = dict[str, Union[Variable, Name]]
CodeStatementWithExpression: TypeAlias = Union[ReturnStatement, Conditional, ExpressionStatement, WhenStatement,
                                               VariableAssignment, AfterStatement, VariableDeclaration]
NameWatchers: TypeAlias = dict[tuple[str, int], tuple[CodeStatementWithExpression, set[tuple[str, int]], list[Namespace], Optional[DreamberdPromise]]]

def get_built_expression(filename: str, code: str, expr: Union[list[Token], ExpressionTreeNode]) -> ExpressionTreeNode:
    return expr if isinstance(expr, ExpressionTreeNode) else build_expression_tree(filename, expr, code)

def get_modified_next_name(name: str, ns: int) -> str:
    return f"{name}_{ns}__next"

def get_modified_prev_name(name: str) -> str:
    return f"{name.replace('.', '__')}__prev"

# i believe this function is exclusively called from the evaluate_expression function
def evaluate_normal_function(filename: str, code: str, expr: FunctionNode, func: Union[DreamberdFunction, BuiltinFunction], namespaces: list[Namespace], async_statements: list[tuple[list[tuple[CodeStatement, ...]], list[Namespace]]], name_watchers: NameWatchers) -> Value:
    args = [evaluate_expression(filename, code, arg, namespaces, async_statements, name_watchers) for arg in expr.args] 

    # check to evaluate builtin
    if isinstance(func, BuiltinFunction):
        if func.arg_count > len(args):
            raise_error_at_token(filename, code, f"Expected more arguments for function call with {func.arg_count} argument{'s' if func.arg_count == 1 else ''}.", expr.name)
        try:
            max_arg_count = func.arg_count if func.arg_count != -1 else len(args)
            return func.function(*args[:max_arg_count]) or DreamberdUndefined()
        except InterpretationError as e:  # some intentionally raised error
            raise_error_at_token(filename, code, str(e), expr.name); raise
    
    # check length is proper, adjust namespace, and run this code
    if len(func.args) > len(args):
        raise_error_at_token(filename, code, f"Expected more arguments for function call with {len(func.args)} argument{'s' if len(func.args) == 1 else ''}.", expr.name)
    new_namespace: Namespace = {name: Name(name, arg) for name, arg in zip(func.args, args)}
    namespaces.append(new_namespace)
    retval = interpret_code_statements(filename, code, func.code, namespaces, async_statements, name_watchers)
    namespaces.pop()

    return retval or DreamberdUndefined()

def register_async_function(filename: str, code: str, expr: FunctionNode, func: DreamberdFunction, namespaces: list[Namespace], async_statements: list[tuple[list[tuple[CodeStatement, ...]], list[Namespace]]], name_watchers: NameWatchers) -> None:
    """ Adds a job to the async statements queue, which is accessed in the interpret_code_statements function. """
    args = [evaluate_expression(filename, code, arg, namespaces, async_statements, name_watchers) for arg in expr.args]
    if len(func.args) < len(args):
        raise_error_at_token(filename, code, f"Expected more arguments for function call with {len(func.args)} argument{'s' if len(func.args) == 1 else ''}.", expr.name)
    function_namespaces = copy(namespaces) + [{name: Name(name, arg) for name, arg in zip(func.args, args)}]
    async_statements.append((func.code, function_namespaces))

def declare_new_variable(filename: str, code: str, name: str, value: Value, lifetime: Optional[str], confidence: int, namespaces: list[Namespace], async_statements: list[tuple[list[tuple[CodeStatement, ...]], list[Namespace]]], name_watchers: NameWatchers):

    if len(name.split('.')) > 1:
        raise InterpretationError("Cannot declare a variable with periods in the name.")

    is_lifetime_temporal = lifetime is not None and not lifetime[-1].isdigit()
    variable_duration = 100000000000 if is_lifetime_temporal or lifetime is None else int(lifetime)
    target_lifetime = VariableLifetime(value, variable_duration, confidence)
    
    if v := namespaces[-1].get(name):
        if isinstance(v, Variable):
            target_var = v
            for i in range(len(v.lifetimes)):
                if v.lifetimes[i].confidence == confidence or i == len(v.lifetimes) - 1:
                    if i == 0:
                        v.prev_values.append(v.value)
                    v.lifetimes[i:i] = [target_lifetime]
        else:
            target_var = Variable(name, [target_lifetime], [v.value])
            namespaces[-1][name] = target_var
    else:  # for loop finished unbroken, no matches found
        target_var = Variable(name, [target_lifetime], [])
        namespaces[-1][name] = target_var

    # check if there is a watcher for this name
    watchers_key = (name, id(namespaces[-1]))
    if y := name_watchers.get(watchers_key):
        mod_name = get_modified_next_name(*watchers_key)
        y[2][-1][mod_name] = Name(mod_name, value)  # add the value to the uppermost namespace
        y[1].remove(watchers_key)                   # remove the name from the set containing remaining names
        if not y[1]:  # not waiting on anybody else, execute the code
            interpret_name_watching_statement(filename, code, y[0], y[2], y[3], async_statements, name_watchers)
        del name_watchers[watchers_key]             # stop watching this name
        
    # if we're dealing with seconds just sleep in another thread and remove the variable lifetime
    if is_lifetime_temporal:
        def remove_lifetime(lifetime: str, target_var: Variable, target_lifetime: VariableLifetime):
            if lifetime[-1] not in ['s', 'm']:
                raise InterpretationError("Invalid time unit for variable lifetime.")
            sleep(int(lifetime[:-1]) if lifetime[-1] == 's' else int(lifetime[:-1] * 60))
            for i, lt in reversed([*enumerate(target_var.lifetimes)]):
                if lt is target_lifetime:
                    del target_var.lifetimes[i]
        Thread(target=remove_lifetime, args=(lifetime, target_var, target_lifetime)).start()

def assign_variable(filename: str, code: str, name: str, index: Optional[Value], value: Value, namespaces: list[Namespace], async_statements: list[tuple[list[tuple[CodeStatement, ...]], list[Namespace]]], name_watchers: NameWatchers):
    pass
        
def get_value_from_promise(val: DreamberdPromise) -> Value:
    if val.value is None:
        return DreamberdUndefined()
    return val.value

def get_name_from_namespaces(val: str, namespaces: list[Namespace]) -> Optional[Union[Variable, Name]]:
    """ This is called when we are sure that the value is a name. """
    if len(v_split := val.split('.')) == 1:
        for ns in namespaces:
            if (v := ns.get(val)) is not None:
                return v
    else:
        base_val = get_name_from_namespaces(v_split[0], namespaces)
        if not base_val:   # base object not found
            return None
        for other_name in v_split[1:]:
            if not (ns := getattr(base_val.value, "namespace")):  # the value does not have a namespace attribute
                return None
            base_val = get_name_from_namespaces(other_name, ns)
            if not base_val:   # the value was not found in the namespace
                return None
        return base_val
    return None

def get_name_and_namespace_from_namespaces(val: str, namespaces: list[Namespace]) -> tuple[Optional[Union[Variable, Name]], Optional[Namespace]]:
    """ This is the same as the function defined above, except it also returns the namespace in which the name was found. """
    if len(v_split := val.split('.')) == 1:
        for ns in namespaces:
            if (v := ns.get(val)) is not None:
                return v, ns
    else:
        base_val, ns = get_name_and_namespace_from_namespaces(v_split[0], namespaces)
        if not base_val:
            return None, ns  # value doesn't exist but the namespace does
        for other_name in v_split[1:]:
            if not (ns := getattr(base_val.value, "namespace")):  # namespace does not exist, nothing to look for
                return None, None
            base_val = get_name_from_namespaces(other_name, ns)
            if not base_val:  # same case as before
                return None, ns
        return base_val, ns
    return None, namespaces[-1]

def determine_non_name_value(val: Token) -> Value:
    """ 
    Takes a string/Token and determines if the value is a number, string, or invalid. 
    Valid names should have been found already by the previous function.
    """
    if len(v := val.value.split('.')) <= 2:
        if all(x.isdigit() for x in v):
            return DreamberdNumber([int, float][len(v) - 1](val.value))
    return DreamberdString(val.value)

def is_approx_equal(left: Value, right: Value) -> DreamberdBoolean:

    if is_really_really_equal(left, right).value:
        return DreamberdBoolean(True)

    if isinstance(left, DreamberdString) or isinstance(right, DreamberdString):

        return DreamberdBoolean(SequenceMatcher(None, db_to_string(left).value, 
                                                db_to_string(right).value).ratio() > STRING_EQUALITY_RATIO)

    if isinstance((num := left), DreamberdNumber) or isinstance((num := right), DreamberdNumber):
        other = left if right == num else right
        if isinstance(other, (DreamberdNumber, DreamberdUndefined, DreamberdBoolean)):
            left_num, right_num = db_to_number(left).value, db_to_number(right).value
            return DreamberdBoolean(left_num == right_num or (False if left_num == 0 else (left_num - right_num) / left_num > NUM_EQUALITY_RATIO))

    if isinstance(left, DreamberdBoolean) or isinstance(right, DreamberdBoolean):
        left_bool, right_bool = db_to_boolean(left).value, db_to_boolean(right).value
        if left_bool is None or right_bool is None:
            return DreamberdBoolean(None)  # maybe
        return DreamberdBoolean(left_bool == right_bool)

    if (val := db_to_boolean(left).value) == db_to_boolean(right).value and val is not None:
        return DreamberdBoolean(True)

    if type(left) != type(right):
        return DreamberdBoolean(None)  # maybe, programmer got too lazy

    if isinstance(left, DreamberdList) and isinstance(right, DreamberdList):
        if len(left.values) == len(right.values) == 0:
            return DreamberdBoolean(True)
        is_equals = [is_approx_equal(l, r) for l, r in zip(left.values, right.values)]
        ratio = sum([int(x.value) if x.value is not None else 0.5 for x in is_equals]) / max(len(left.values), len(right.values))
        return DreamberdBoolean(ratio > LIST_EQUALITY_RATIO)

    if isinstance(left, DreamberdMap) and isinstance(right, DreamberdMap):
        if len(left.self_dict) == len(right.self_dict) == 0:
            return DreamberdBoolean(True)
        is_equals = [is_approx_equal(left.self_dict[key], right.self_dict[key])
                     for key in left.self_dict.keys() & right.self_dict.keys()]
        ratio = sum([int(x.value) if x.value is not None else 0.5 for x in is_equals]) /                  \
                len(left.self_dict.keys() | right.self_dict.keys())
        return DreamberdBoolean(ratio > MAP_EQUALITY_RATIO)

    if isinstance(left, DreamberdFunction) and isinstance(right, DreamberdFunction):
        if len(left.code) == len(right.code) == 0:
            return DreamberdBoolean(True)
        ratio = sum([len(set(l) | set(r)) / min(len(l), len(r)) for l, r in zip(left.code, right.code)]) / max(len(left.code), len(right.code))
        return DreamberdBoolean(True if ratio > FUNCTION_EQUALITY_RATIO else None)  # for no reason whatsoever, this will be maybe and not False

    if isinstance(left, DreamberdObject) and isinstance(right, DreamberdObject):
        if len(left.namespace) == len(right.namespace) == 0:
            return DreamberdBoolean(True)
        is_equals = [is_approx_equal(left.namespace[key].value, right.namespace[key].value)
                     for key in left.namespace.keys() & right.namespace.keys()]
        ratio = sum([int(x.value) if x.value is not None else 0.5 for x in is_equals]) /                  \
                len(left.namespace.keys() | right.namespace.keys())
        return DreamberdBoolean(ratio > OBJECT_EQUALITY_RATIO)

    return DreamberdBoolean(None)

def is_equal(left: Value, right: Value) -> DreamberdBoolean:

    if isinstance(left, DreamberdString) or isinstance(right, DreamberdString):
        return DreamberdBoolean(db_to_string(left).value == db_to_string(right).value)

    if isinstance(left, DreamberdNumber) or isinstance(right, DreamberdNumber):
        return DreamberdBoolean(db_to_number(left).value == db_to_number(right).value)

    if isinstance(left, DreamberdBoolean) or isinstance(right, DreamberdBoolean):
        left_bool, right_bool = db_to_boolean(left).value, db_to_boolean(right).value
        if left_bool is None or right_bool is None:
            return DreamberdBoolean(None)  # maybe
        return DreamberdBoolean(left_bool == right_bool)

    if (val := db_to_boolean(left).value) == db_to_boolean(right).value and val is not None:
        return DreamberdBoolean(True)

    if type(left) != type(right):
        return DreamberdBoolean(None)  # maybe, programmer got too lazy

    if isinstance(left, DreamberdList) and isinstance(right, DreamberdList):
        return DreamberdBoolean(all([is_equal(l, r).value for l, r in zip(left.values, right.values)]))

    if isinstance(left, DreamberdMap) and isinstance(right, DreamberdMap):
        is_equals = [is_approx_equal(left.self_dict[key], right.self_dict[key]).value 
                     for key in left.self_dict.keys() & right.self_dict.keys()]
        return DreamberdBoolean(all(is_equals))

    if isinstance(left, DreamberdObject) and isinstance(right, DreamberdObject):
        is_equals = [is_approx_equal(left.namespace[key].value, right.namespace[key].value).value 
                     for key in left.namespace.keys() & right.namespace.keys()]
        return DreamberdBoolean(all(is_equals))

    return DreamberdBoolean(None)

def is_really_equal(left: Value, right: Value) -> DreamberdBoolean:
    if type(left) != type(right):
        return DreamberdBoolean(False)
    match left:
        case DreamberdNumber() | DreamberdString() | DreamberdBoolean() | DreamberdKeyword():
            return DreamberdBoolean(left.value == right.value)
        case DreamberdUndefined():
            return DreamberdBoolean(True) 
        case DreamberdObject():
            return DreamberdBoolean(left.class_name == right.class_name and 
                                    left.namespace.keys() == right.namespace.keys() and 
                                    all([is_really_equal(left.namespace[k].value, right.namespace[k].value).value for k in left.namespace.keys()]))
        case DreamberdFunction():
            return DreamberdBoolean(all([getattr(left, name) == getattr(right, name) for name in ["code", "args", "is_async"]]))
        case DreamberdList():
            return DreamberdBoolean(len(left.values) == len(right.values) and 
                                    all([is_really_equal(l, r).value for l, r in zip(left.values, right.values)]))
        case DreamberdMap():
            return DreamberdBoolean(left.self_dict.keys() == right.self_dict.keys() and 
                                    all([is_really_equal(left.self_dict[k], right.self_dict[k]).value for k in left.self_dict]))
    return DreamberdBoolean(None)

def is_really_really_equal(left: Value, right: Value) -> DreamberdBoolean:
    return DreamberdBoolean(left is right)

def is_less_than(left: Value, right: Value) -> DreamberdBoolean:
    if type(left) != type(right):
        raise InterpretationError('Cannot compare two values of different types.')
    match left:
        case DreamberdNumber() | DreamberdString() | DreamberdBoolean():
            if isinstance(left, DreamberdBoolean) and (left.value is None or right.value is None):
                return DreamberdBoolean(None)
            return DreamberdBoolean(left.value < right.value)
        case DreamberdUndefined():
            return DreamberdBoolean(False)
        case DreamberdList():
            return DreamberdBoolean(len(left.values) < len(right.values))
        case DreamberdMap():
            return DreamberdBoolean(len(left.self_dict) < len(right.self_dict))
        case DreamberdKeyword() | DreamberdObject() | DreamberdFunction():
            raise InterpretationError(f"Comparison not supported between elements of type {type(left).__name__}.")
    return DreamberdBoolean(None)

def perform_single_value_operation(filename: str, code: str, expr: Value, operator: OperatorType, operator_token: Token) -> Value: 
    pass

def perform_two_value_operation(filename: str, code: str, left: Value, right: Value, operator: OperatorType, operator_token: Token) -> Value:
    try:
        match operator:
            case OperatorType.ADD:
                if isinstance(left, DreamberdString) or isinstance(right, DreamberdString):
                    return DreamberdString(db_to_string(left).value + db_to_string(right).value)
                left_num = db_to_number(left)
                right_num = db_to_number(right)
                return DreamberdNumber(left_num.value + right_num.value)
            case OperatorType.SUB | OperatorType.MUL | OperatorType.DIV | OperatorType.EXP:
                left_num = db_to_number(left)
                right_num = db_to_number(right)
                if operator == OperatorType.DIV and abs(right_num.value) < FLOAT_TO_INT_PREC: # pretty much zero
                    return DreamberdUndefined() 
                elif operator == OperatorType.EXP and left_num.value < -FLOAT_TO_INT_PREC and not is_int(right_num.value):
                    raise InterpretationError("Cannot raise a negative base to a non-integer exponent.")
                match operator:
                    case OperatorType.SUB: result = left_num.value - right_num.value
                    case OperatorType.MUL: result = left_num.value * right_num.value
                    case OperatorType.DIV: result = left_num.value / right_num.value
                    case OperatorType.EXP: result = pow(left_num.value, right_num.value)
                return DreamberdNumber(result)
            case OperatorType.OR:
                left_bool = db_to_boolean(left)
                right_bool = db_to_boolean(right) 
                match left_bool.value, right_bool.value:
                    case True, _:     return left    # yes 
                    case False, _:    return right   # depends
                    case None, True:  return right   # yes
                    case None, False: return left    # maybe?
                    case None, None:  return left if random.random() < 0.50 else right   # maybe? 
            case OperatorType.AND:  
                left_bool = db_to_boolean(left)
                right_bool = db_to_boolean(right) 
                match left_bool.value, right_bool.value:
                    case True, _:     return right   # depends
                    case False, _:    return left    # nope
                    case None, True:  return left    # maybe?
                    case None, False: return right   # nope
                    case None, None:  return left if random.random() < 0.50 else right  # maybe? 
            case OperatorType.E: 
                return is_approx_equal(left, right)

            # i'm gonna call this lasagna code because it's stacked like lasagna and looks stupid
            case OperatorType.EE | OperatorType.NE:
                if operator == OperatorType.EE:
                    return is_equal(left, right)
                return db_not(is_equal(left, right))
            case OperatorType.EEE | OperatorType.NEE:
                if operator == OperatorType.EEE:
                    return is_really_equal(left, right)
                return db_not(is_really_equal(left, right))
            case OperatorType.EEEE | OperatorType.NEEE:
                if operator == OperatorType.EEEE:
                    return is_really_really_equal(left, right)
                return db_not(is_really_really_equal(left, right))
            case OperatorType.GE | OperatorType.LE:
                if is_really_equal(left, right):
                    return DreamberdBoolean(True)
                if operator == OperatorType.LE:
                    return is_less_than(left, right)
                return db_not(is_less_than(left, right))
            case OperatorType.LT, OperatorType.GT:
                if operator == OperatorType.LT:
                    return is_less_than(left, right)
                return db_not(is_less_than(left, right))

    except InterpretationError as e:
        raise_error_at_token(filename, code, str(e), operator_token)  # the operator token is the best that you are gonna get
    raise_error_at_token(filename, code, "Something went wrong here.", operator_token); raise

def evaluate_expression(filename: str, code: str, expr: ExpressionTreeNode, namespaces: list[dict[str, Union[Variable, Name]]], async_statements: list[tuple[list[tuple[CodeStatement, ...]], list[Namespace]]], name_watchers: NameWatchers) -> Value:

    match expr:
        case FunctionNode():  # done :)
            
            # for a function, the thing must be in the namespace
            func = get_name_from_namespaces(expr.name.value, namespaces)

            # make sure it exists and it is actually a function in the namespace
            if func is None:
                raise_error_at_token(filename, code, "Cannot find token in namespace.", expr.name); raise
    
            # check the thing in the await symbol. if awaiting a single function that is async, evaluate it as not async
            force_execute_sync = False
            if isinstance(func.value, DreamberdKeyword) and func.value.value == "await":
                if len(expr.args) != 1:
                    raise_error_at_token(filename, code, "Expected only one argument for await function.", expr.name); raise
                if not isinstance(expr.args[0], FunctionNode):
                    raise_error_at_token(filename, code, "Expected argument of await function to be a function call.", expr.name); raise
                force_execute_sync = True 
                
                # check for None again
                expr = expr.args[0]
                func = get_name_from_namespaces(expr.name.value, namespaces)
                if func is None:  # the other check happens in the next statement
                    raise_error_at_token(filename, code, "Cannot find token in namespaces.", expr.name); raise

            if not isinstance(func.value, (BuiltinFunction, DreamberdFunction)):
                raise_error_at_token(filename, code, "Attempted function call on non-function value.", expr.name); raise
            
            if isinstance(func.value, DreamberdFunction) and func.value.is_async and not force_execute_sync:
                register_async_function(filename, code, expr, func.value, namespaces, async_statements, name_watchers)
                return DreamberdUndefined()
            return evaluate_normal_function(filename, code, expr, func.value, namespaces, async_statements, name_watchers)

        case ListNode():  # done :) 
            return DreamberdList([evaluate_expression(filename, code, x, namespaces, async_statements, name_watchers) for x in expr.values])
        case ValueNode():  # done :)

            # what the fuck am i doing rn
            if v := get_name_from_namespaces(expr.name_or_value.value, namespaces):
                if isinstance(v.value, DreamberdPromise):
                    return deepcopy(get_value_from_promise(v.value))
                return deepcopy(v.value)  # nothing is mutable - CHANGEABLE
            return determine_non_name_value(expr.name_or_value)

        case IndexNode():  # not done :(
            pass
        case ExpressionNode():  # done :)
            left = evaluate_expression(filename, code, expr.left, namespaces, async_statements, name_watchers)
            right = evaluate_expression(filename, code, expr.right, namespaces, async_statements, name_watchers)
            return perform_two_value_operation(filename, code, left, right, expr.operator, expr.operator_token)
    return DreamberdUndefined()

def handle_next_expressions(filename: str, code: str, expr: ExpressionTreeNode, namespaces: list[Namespace]) -> tuple[ExpressionTreeNode, set[tuple[str, int]], set[str]]:

    """ 
    This function looks for the "next" keyword in an expression, and detects seperate await modifiers for that keyword.
    Then, it removes the "next" and "await next" nodes from the ExpressionTree, and returns the head of the tree.
    Additionally, every name that appears in the function as a next or async next, its value is saved in a temporary namespace.
    With the returned set of names that are used in "next" and "await next", we can insert these into a dictionary
        that contains information about which names are being "watched" for changes. When this dictionary changes,
        we can execute code accordingly.
    """

    normal_nexts: set[tuple[str, int]] = set()
    async_nexts: set[str] = set()
    inner_nexts = []
    match expr:
        case FunctionNode():

            func = get_name_from_namespaces(expr.name.value, namespaces)
            if func is None:
                raise_error_at_token(filename, code, "Attempted function call on undefined variable.", expr.name); raise

            # check if it is a next or await 
            if isinstance(func.value, DreamberdKeyword) and \
               ((is_next := func.value.value == "next") or (is_await := func.value.value == "await")):

                if is_next:

                    # add it to list of things to watch for and change the returned expression to the name being next-ed
                    if len(expr.args) != 1 or not isinstance(expr.args[0], ValueNode):
                        raise_error_at_token(filename, code, "\"Next\"keyword can only take a single value as an argument.", expr.name); raise
                    name = expr.args[0].name_or_value.value
                    _, ns = get_name_and_namespace_from_namespaces(name, namespaces)
                    if not ns:
                        raise InterpretationError("Attempted to access namespace of a value without a namespace.")
                    last_name = name.split('.')[-1]
                    normal_nexts.add((name, id(ns)))
                    expr = expr.args[0]
                    expr.name_or_value.value = get_modified_next_name(last_name, id(ns))

                elif is_await:

                    if len(expr.args) != 1 or not isinstance(expr.args[0], FunctionNode):
                        raise_error_at_token(filename, code, "Can only await a function.", expr.name); raise
                    inner_expr = expr.args[0]
                        
                    func = get_name_from_namespaces(expr.args[0].name.value, namespaces)
                    if func is None:
                        raise_error_at_token(filename, code, "Attempted function call on undefined variable.", expr.name); raise

                    if isinstance(func.value, DreamberdKeyword) and func.value.value == "next":
                        if len(inner_expr.args) != 1 or not isinstance(inner_expr.args[0], ValueNode):
                            raise_error_at_token(filename, code, "\"Next\"keyword can only take a single value as an argument.", inner_expr.name); raise
                        name = inner_expr.args[0].name_or_value.value 
                        _, ns = get_name_and_namespace_from_namespaces(name, namespaces)
                        if not ns:
                            raise InterpretationError("Attempted to access namespace of a value without a namespace.")
                        last_name = name.split('.')[-1]
                        async_nexts.add(name)  # only need to store the name for the async ones because we are going to wait anyways
                        expr = inner_expr.args[0]
                        expr.name_or_value.value = get_modified_next_name(last_name, id(ns))

            else:
                replacement_args = []
                for arg in expr.args:
                    new_expr, *nexts = handle_next_expressions(filename, code, arg, namespaces)
                    inner_nexts.append(nexts)
                    replacement_args.append(new_expr)
                expr.args = replacement_args
                
        case ListNode():
            replacement_values = []
            for ex in expr.values:
                new_expr, *nexts = handle_next_expressions(filename, code, ex, namespaces)
                inner_nexts.append(nexts)
                replacement_values.append(new_expr)
            expr.values = replacement_values
        case IndexNode():
            new_value, *value_nexts = handle_next_expressions(filename, code, expr.value, namespaces)
            new_index, *index_nexts = handle_next_expressions(filename, code, expr.index, namespaces)
            expr.value = new_value 
            expr.index = new_index
            inner_nexts.extend([value_nexts, index_nexts])
        case ExpressionNode():
            new_left, *left_nexts = handle_next_expressions(filename, code, expr.left, namespaces)
            new_right, *right_nexts = handle_next_expressions(filename, code, expr.right, namespaces)
            expr.left = new_left 
            expr.right = new_right
            inner_nexts.extend([left_nexts, right_nexts])
    for nn, an in inner_nexts:
        normal_nexts |= nn 
        async_nexts |= an 
    return expr, normal_nexts, async_nexts

def save_previous_values_next_expr(filename: str, code: str, expr_to_modify: ExpressionTreeNode, nexts: set[str], namespaces: list[Namespace]) -> Namespace:

    saved_namespace: Namespace = {}
    match expr_to_modify:
        case ValueNode():
            name = expr_to_modify.name_or_value.value
            val = get_name_from_namespaces(name, namespaces)
            if not val:
                raise InterpretationError("Attempting to access undefined variable.")
            mod_name = get_modified_prev_name(name)
            saved_namespace[mod_name] = Name(mod_name, val.value)
        case ExpressionNode():
            left_ns = save_previous_values_next_expr(filename, code, expr_to_modify.left, nexts, namespaces)
            right_ns = save_previous_values_next_expr(filename, code, expr_to_modify.right, nexts, namespaces)
            saved_namespace |= left_ns | right_ns
        case IndexNode():
            value_ns = save_previous_values_next_expr(filename, code, expr_to_modify.value, nexts, namespaces)
            index_ns = save_previous_values_next_expr(filename, code, expr_to_modify.index, nexts, namespaces)
            saved_namespace |= value_ns | index_ns
        case ListNode():
            for ex in expr_to_modify.values:
                ns = save_previous_values_next_expr(filename, code, ex, nexts, namespaces)
                saved_namespace |= ns
        case FunctionNode():
            for arg in expr_to_modify.args:
                ns = save_previous_values_next_expr(filename, code, arg, nexts, namespaces)
                saved_namespace |= ns
    return saved_namespace

def determine_statement_type(possible_statements: tuple[CodeStatement, ...], namespaces: list[Namespace]) -> Optional[CodeStatement]:
    instance_to_keywords: dict[type[CodeStatement], set[str]] = {
        Conditional: {'if'},
        WhenStatement: {'when'},
        AfterStatement: {'after'},
        ClassDeclaration: {'class', 'className'},
        ReturnStatement: {'return'},
        DeleteStatement: {'delete'}
    }
    for st in possible_statements:
        if kw := instance_to_keywords.get(type(st)):
            val = get_name_from_namespaces(st.keyword, namespaces)
            if val is not None and isinstance(val.value, DreamberdKeyword) and val.value.value in kw:
                return st
        elif isinstance(st, FunctionDefinition):  # allow for async and normal function definitions
            if len(st.keywords) == 1:
                val = get_name_from_namespaces(st.keywords[0], namespaces)
                if val and isinstance(val.value, DreamberdKeyword) and re.match(r"f?u?n?c?t?i?o?n?", val.value.value):
                    return st
            elif len(st.keywords) == 2:
                val = get_name_from_namespaces(st.keywords[0], namespaces)
                other_val = get_name_from_namespaces(st.keywords[1], namespaces)
                if val and other_val and isinstance(val.value, DreamberdKeyword) and isinstance(other_val.value, DreamberdKeyword) \
                   and re.match(r"f?u?n?c?t?i?o?n?", val.value.value) and other_val.value.value == 'async':
                    return st
        elif isinstance(st, VariableDeclaration):  # allow for const const const and normal declarations
            if len(st.modifiers) == 2:
                if all([(val := get_name_from_namespaces(mod, namespaces)) is not None and 
                    isinstance(val.value, DreamberdKeyword) and val.value.value in {'const', 'var'}
                    for mod in st.modifiers]):
                    return st
            elif len(st.modifiers) == 3:
                if all([(val := get_name_from_namespaces(mod, namespaces)) is not None and 
                    isinstance(val.value, DreamberdKeyword) and val.value.value == 'const' 
                    for mod in st.modifiers]):
                    return st

    # now is left: expression evalulation and variable assignment
    for st in possible_statements:
        if isinstance(st, VariableAssignment):
            return st 
    for st in possible_statements:
        if isinstance(st, ExpressionStatement):
            return st
    return None 

def adjust_for_normal_nexts(statement: CodeStatementWithExpression, async_nexts: set[str], normal_nexts: set[tuple[str, int]], promise: Optional[DreamberdPromise], namespaces: list[Namespace], name_watchers: NameWatchers):

    old_async_vals, old_normal_vals = [], []
    get_state_watcher = lambda val: None if not val else len(v) if (v := getattr(val, "prev_values")) else 0
    for name in async_nexts:
        old_async_vals.append(get_state_watcher(get_name_from_namespaces(name, namespaces))) 
    for name, _ in normal_nexts:
        old_normal_vals.append(get_state_watcher(get_name_from_namespaces(name, namespaces)))

    # for each async one, wait until each one is different
    for name, start_len in zip(async_nexts, old_async_vals): 
        curr_len = get_state_watcher(get_name_from_namespaces(name, namespaces))
        while start_len != curr_len:  
            curr_len = get_state_watcher(get_name_from_namespaces(name, namespaces))

    # now, build a namespace for each one
    new_namespace: Namespace = {}
    for name, old_len in zip(async_nexts, old_async_vals):
        v, ns = get_name_and_namespace_from_namespaces(name, namespaces)
        if not v or not ns or (old_len is not None and not isinstance(v, Variable)):
            raise InterpretationError("Something went wrong with accessing the next value of a variable.")
        mod_name = get_modified_next_name(name, id(ns))
        match old_len:
            case None: new_namespace[mod_name] = Name(mod_name, v.value if isinstance(v, Name) else v.prev_values[0])
            case i:
                if not isinstance(v, Variable):
                    raise InterpretationError("Something went wrong.")
                new_namespace[mod_name] = Name(mod_name, v.prev_values[i])

    # now, adjust for any values that may have already been modified by next statements 
    for (name, ns_id), old_len in zip(normal_nexts, old_normal_vals):
        new_len = get_state_watcher(v := get_name_from_namespaces(name, namespaces))
        if v is None or new_len == old_len: 
            continue
        mod_name = get_modified_next_name(name, ns_id)
        normal_nexts.remove((name, ns_id))
        match old_len:
            case None: new_namespace[mod_name] = Name(mod_name, v.value if isinstance(v, Name) else v.prev_values[0])
            case i:
                if not isinstance(v, Variable):
                    raise InterpretationError("Something went wrong.")
                new_namespace[mod_name] = Name(mod_name, v.prev_values[i])

    # the remaining values are still waiting on a result, add these to the list of name watchers
    # this new_namespace i am adding is purely for use in evaluation of expressions, and the code within 
    # a code statement should not use that namespace. therefore, some logic must be done to remove that namespace 
    # when the expression of a code statement is executed
    name_watchers |= {(name.split('.')[-1], ns_id): (statement, normal_nexts, namespaces + [new_namespace], promise) for name, ns_id in normal_nexts} 

def wait_for_async_nexts(async_nexts: set[str], namespaces: list[Namespace]) -> Namespace:

    old_async_vals = []
    get_state_watcher = lambda val: None if not val else len(v) if (v := getattr(val, "prev_values")) else 0
    for name in async_nexts:
        old_async_vals.append(get_state_watcher(get_name_from_namespaces(name, namespaces))) 

    # for each async one, wait until each one is different
    for name, start_len in zip(async_nexts, old_async_vals): 
        curr_len = get_state_watcher(get_name_from_namespaces(name, namespaces))
        while start_len != curr_len:  
            curr_len = get_state_watcher(get_name_from_namespaces(name, namespaces))

    # now, build a namespace for each one
    new_namespace: Namespace = {}
    for name, old_len in zip(async_nexts, old_async_vals):
        v, ns = get_name_and_namespace_from_namespaces(name, namespaces)
        if not v or not ns or (old_len is not None and not isinstance(v, Variable)):
            raise InterpretationError("Something went wrong with accessing the next value of a variable.")
        mod_name = get_modified_next_name(name, id(ns))
        match old_len:
            case None: new_namespace[mod_name] = Name(mod_name, v.value if isinstance(v, Name) else v.prev_values[0])
            case i:
                if not isinstance(v, Variable):
                    raise InterpretationError("Something went wrong.")
                new_namespace[mod_name] = Name(mod_name, v.prev_values[i])
    return new_namespace

def interpret_name_watching_statement(filename, code, statement: CodeStatementWithExpression, namespaces: list[Namespace], promise: Optional[DreamberdPromise], async_statements: list[tuple[list[tuple[CodeStatement, ...]], list[Namespace]]], name_watchers: NameWatchers): 
    # evaluate the expression using the names off the top
    expr_val = evaluate_expression(filename, code, get_built_expression(filename, code, statement.expression), namespaces, async_statements, name_watchers) 
    namespaces.pop()  # remove expired namespace
    match statement:
        case ReturnStatement():
            if promise is None:
                raise InterpretationError("Something went wrong.")
            promise.value = expr_val  # simply change the promise to that value as the return statement already returned a promise
        case VariableDeclaration():
            declare_new_variable(filename, code, statement.name, expr_val, statement.lifetime, statement.confidence, namespaces, async_statements, name_watchers)
        case VariableAssignment():
            pass
        case Conditional():
            pass 
        case AfterStatement():
            pass 
        case ExpressionStatement():  # literally no action required because the expression was already evaluated
            pass

def clear_temp_namespace(namespaces: list[Namespace], temp_namespace: Namespace) -> None:
    for key in temp_namespace:
        del namespaces[-1][key]

def interpret_statement(filename: str, code: str, statement: CodeStatement, namespaces: list[Namespace], async_statements:  list[tuple[list[tuple[CodeStatement, ...]], list[Namespace]]], name_watchers: NameWatchers):

 
    # build a list of expressions that are modified to allow for the next keyword
    expressions_to_check: list[Union[list[Token], ExpressionTreeNode]] = []
    match statement:
        case VariableAssignment(): expressions_to_check = [statement.expression] + statement.index
        case VariableDeclaration() | Conditional() | WhenStatement() | AfterStatement() | ExpressionStatement(): expressions_to_check = [statement.expression]
    all_normal_nexts: set[tuple[str, int]] = set()
    all_async_nexts: set[str] = set()
    next_filtered_exprs = []
    for expr in expressions_to_check:
        built_expr = get_built_expression(filename, code, expr)
        new_expr, normal_nexts, async_nexts = handle_next_expressions(filename, code, built_expr, namespaces)
        all_normal_nexts |= normal_nexts 
        all_async_nexts |= async_nexts
        next_filtered_exprs.append(new_expr)
    prev_namespace: Namespace = {}
    all_nexts = {s[0] for s in all_normal_nexts} | all_async_nexts
    for expr in next_filtered_exprs:
        prev_namespace |= save_previous_values_next_expr(filename, code, expr, all_nexts, namespaces)
    namespaces[-1] |= prev_namespace
    if all_normal_nexts:
        match statement:
            case VariableAssignment():
                statement.expression = next_filtered_exprs[0]
                statement.index = next_filtered_exprs[1:]
            case VariableDeclaration() | Conditional() | WhenStatement() | AfterStatement() | ExpressionStatement():
                statement.expression = next_filtered_exprs[0]
            case _:
                raise InterpretationError("Something went wrong. It's not your fault, it's mine.")
        adjust_for_normal_nexts(statement, all_async_nexts, all_normal_nexts, None, namespaces, name_watchers)
    else:
        if all_async_nexts:
            prev_namespace |= wait_for_async_nexts(all_async_nexts, namespaces)
            namespaces[-1] |= prev_namespace
        # perform statement matching here
    clear_temp_namespace(namespaces, prev_namespace)

    match statement:
        
        case VariableAssignment():
            evaled_expression = get_built_expression(filename, code, statement.expression)
            evaled_index_expressions = [get_built_expression(filename, code, x) for x in statement.index] if statement.index else []
            expr, normal_nexts, async_nexts = handle_next_expressions(filename, code, evaled_expression, namespaces)
            index_exprs = []
            for index_expr in evaled_index_expressions:
                index_expr, index_normal_nexts, index_async_nexts = handle_next_expressions(filename, code, index_expr, namespaces)
                normal_nexts |= index_normal_nexts
                async_nexts |= index_async_nexts
                index_exprs.append(index_expr)
            all_next_names = {s[0] for s in normal_nexts} | async_nexts
            temp_namespace = save_previous_values_next_expr(filename, code, expr, all_next_names, namespaces)
            for index_expr in index_exprs:
                temp_namespace |= save_previous_values_next_expr(filename, code, index_expr, all_next_names, namespaces)
            if normal_nexts:
                adjust_for_normal_nexts(statement, expr, async_nexts, normal_nexts, None, namespaces, name_watchers)
            else:
                if async_nexts:
                    temp_namespace = wait_for_async_nexts(async_nexts, namespaces)
                assign_variable(filename, code, statement.name, statement.index, evaluate_expression(
                    filename, code, expr, namespaces, async_statements, name_watchers
                ), namespaces, async_statements, name_watchers)
        case DeleteStatement():
            pass

        case ExpressionStatement():
            pass

        case Conditional():
            pass

        case AfterStatement():
            pass

        case WhenStatement():
            pass

        case FunctionDefinition():
            namespaces[-1][statement.name] = Name(statement.name, DreamberdFunction(
                args = statement.args,
                code = statement.code,
                is_async = statement.is_async
            ))

        case VariableDeclaration():
            evaled_expression = get_built_expression(filename, code, statement.expression)
            expr, normal_nexts, async_nexts = handle_next_expressions(filename, code, evaled_expression, namespaces)
            temp_namespace = save_previous_values_next_expr(filename, code, expr, async_nexts | {s[0] for s in normal_nexts}, namespaces) 
            namespaces[-1] |= temp_namespace
            if normal_nexts:
                # in this case, temp_namespace will only have the data from the previous values
                adjust_for_normal_nexts(statement, expr, async_nexts, normal_nexts, None, namespaces, name_watchers)
            else:
                temp_namespace |= wait_for_async_nexts(async_nexts, namespaces)
                namespaces[-1] |= temp_namespace
                declare_new_variable(filename, code, statement.name, evaluate_expression(
                    filename, code, expr, namespaces, async_statements, name_watchers
                ), statement.lifetime, statement.confidence, namespaces, async_statements, name_watchers)
            clear_temp_namespace(namespaces, temp_namespace)
            
        case ClassDeclaration(): 

            class_namespace: Namespace = {}
            fill_class_namespace(filename, code, statement.code, namespaces, class_namespace, async_statements, name_watchers)

            # create a builtin function that is a closure returning an object of that class
            instance_made = False
            class_name = statement.name

            def class_object_closure(*args: Value) -> DreamberdObject:
                nonlocal instance_made, class_namespace, class_name
                if instance_made:
                    raise InterpretationError(f"Already made instance of the class \"{class_name}\".")
                instance_made = True 
                if constructor := class_namespace.get(class_name):
                    if not isinstance(func := constructor.value, DreamberdFunction):
                        raise InterpretationError("Cannot create class variable with the same name as the class.")
                    if len(func.args) > len(args):
                        raise InterpretationError(f"Expected more arguments for function call with {len(func.args)} argument{'s' if len(func.args) == 1 else ''}.")
                    new_namespace: Namespace = {name: Name(name, arg) for name, arg in zip(func.args, args)}
                    namespaces.append(new_namespace)
                    interpret_code_statements(filename, code, func.code, namespaces, async_statements, name_watchers)
                    namespaces.pop()
                    del class_namespace[class_name]  # remove the constructor
                return DreamberdObject(class_name, class_namespace)

            namespaces[-1][statement.name] = Name(statement.name, BuiltinFunction(-1, class_object_closure))
 
def fill_class_namespace(filename: str, code: str, statements: list[tuple[CodeStatement, ...]], namespaces: list[Namespace], class_namespace: Namespace, async_statements: list[tuple[list[tuple[CodeStatement, ...]], list[Namespace]]], name_watchers: NameWatchers) -> None:
    for possible_statements in statements:
        statement = determine_statement_type(possible_statements, namespaces)
        match statement:
            case FunctionDefinition():
                if "this" in statement.args:
                    raise InterpretationError("\"this\" keyword not allowed in class function declaration arguments.")
                class_namespace[statement.name] = Name(statement.name, DreamberdFunction(
                    args = ["this"] + statement.args,
                    code = statement.code,
                    is_async = statement.is_async
                ))
            case VariableDeclaration(): 

                # why the fuck are my function calls so long i really need some globals  
                var_expr = evaluate_expression(filename, code, get_built_expression(filename, code, statement.expression), namespaces, async_statements, name_watchers)
                declare_new_variable(filename, code, statement.name, var_expr, statement.lifetime, statement.confidence, [class_namespace], async_statements, name_watchers)
            case _:
                raise InterpretationError(f"Unexpected statement of type {type(statement).__name__} in class declaration.")

# if a return statement is found, this will return the expression evaluated at the return. otherwise, it will return None
# this is done to allow this function to be called when evaluating dreamberd functions
def interpret_code_statements(filename: str, code: str, statements: list[tuple[CodeStatement, ...]], namespaces: list[Namespace], async_statements: list[tuple[list[tuple[CodeStatement, ...]], list[Namespace]]], name_watchers: NameWatchers) -> Optional[Value]:

    curr = 0
    while curr < len(statements): 
        statement = determine_statement_type(statements[curr], namespaces)

        # if no statement found -i.e. the user did something wrong
        if statement is None:
            raise InterpretationError("Error parsing statement. Try again.")
        elif isinstance(statement, ReturnStatement):  # if working with a return statement, either return a promise or a value
            expr, normal_nexts, async_nexts = handle_next_expressions(filename, code, get_built_expression(filename, code, statement.expression), namespaces)
            if normal_nexts:
                promise = DreamberdPromise(None)
                adjust_for_normal_nexts(statement, expr, async_nexts, normal_nexts, promise, namespaces, name_watchers)
                return promise
            elif async_nexts:
                wait_for_async_nexts(async_nexts, namespaces)
            return evaluate_expression(filename, code, expr, namespaces, async_statements, name_watchers)
        interpret_statement(filename, code, statement, namespaces, async_statements, name_watchers)

        # emulate taking turns running all the "async" functions
        for async_st, async_ns in async_statements:  
            if not async_st:
                continue
            statement = determine_statement_type(async_st.pop(0), async_ns)
            if statement is None:
                raise InterpretationError("Error parsing statement. Try again.")
            elif isinstance(statement, ReturnStatement):
                raise InterpretationError("Function executed asynchronously cannot have a return statement.")
            interpret_statement(filename, code, statement, async_ns, async_statements, name_watchers)
             
def main(filename: str, code: str, statements: list[tuple[CodeStatement, ...]]) -> None:  # idk what else to call this
    namespace = KEYWORDS.copy()
    interpret_code_statements(filename, code, statements, [namespace], [])

