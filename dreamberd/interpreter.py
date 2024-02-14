# a note about this file: i've appended a "; raise" at the end of nearly every custom raise_error_at_token call, 
# because pyright (my nvim LSP) doesn't recognize the code terminating at the raise_error_at_token, so I add this 
# to make sure it recognizes that and doesn't yell at me because I don't like being yelled at

# thanks to Indently, I have now discovered the NoReturn type, and pyright shuts up :))))))))))))))))

from __future__ import annotations
import os
import re
import locale
import random
import pickle
import requests
from time import sleep
from pathlib import Path
from copy import deepcopy
from threading import Thread
from difflib import SequenceMatcher
from typing import Literal, Optional, TypeAlias, Union 

KEY_MOUSE_IMPORTED = True
try: 
    from pynput import keyboard, mouse
except ImportError:
    KEY_MOUSE_IMPORTED = False

GITHUB_IMPORTED = True 
try: 
    import github 
except ImportError:
    GITHUB_IMPORTED = False

from dreamberd.base import NonFormattedError, OperatorType, Token, TokenType, debug_print, debug_print_no_token, raise_error_at_line, raise_error_at_token
from dreamberd.builtin import FLOAT_TO_INT_PREC, BuiltinFunction, DreamberdBoolean, DreamberdFunction, DreamberdIndexable, DreamberdKeyword, DreamberdList, DreamberdMap, DreamberdMutable, DreamberdNamespaceable, DreamberdNumber, DreamberdObject, DreamberdPromise, DreamberdSpecialBlankValue, DreamberdString, DreamberdUndefined, Name, Variable, Value, VariableLifetime, db_not, db_to_boolean, db_to_number, db_to_string, is_int
from dreamberd.processor.lexer import tokenize as db_tokenize
from dreamberd.processor.expression_tree import ExpressionTreeNode, FunctionNode, ListNode, SingleOperatorNode, ValueNode, IndexNode, ExpressionNode, build_expression_tree, get_expr_first_token
from dreamberd.processor.syntax_tree import AfterStatement, ClassDeclaration, CodeStatement, CodeStatementKeywordable, Conditional, DeleteStatement, ExportStatement, ExpressionStatement, FunctionDefinition, ImportStatement, ReturnStatement, ReverseStatement, VariableAssignment, VariableDeclaration, WhenStatement

# several "ratios" used in the approx equal function
NUM_EQUALITY_RATIO = 0.1  # a-b / b 
STRING_EQUALITY_RATIO = 0.7  # min ratio to be considered equal
LIST_EQUALITY_RATIO = 0.7  # min ratio of all the elements of a list to be equal for the lists to be equal
MAP_EQUALITY_RATIO = 0.6  # lower thresh cause i feel like it
FUNCTION_EQUALITY_RATIO = 0.6  # yeah 
OBJECT_EQUALITY_RATIO = 0.6 

# thing used in the .dreamberd_runtime file
DB_RUNTIME_PATH = ".dreamberd_runtime"
INF_VAR_PATH = ".inf_vars"
INF_VAR_VALUES_PATH = ".inf_vars_values"
DB_VAR_TO_VALUE_SEP = ";;;"  # i'm feeling fancy

# :D 
Namespace: TypeAlias = dict[str, Union[Variable, Name]]
CodeStatementWithExpression: TypeAlias = Union[ReturnStatement, Conditional, ExpressionStatement, WhenStatement,
                                               VariableAssignment, AfterStatement, VariableDeclaration]
AsyncStatements: TypeAlias = list[tuple[list[tuple[CodeStatement, ...]], list[Namespace], int, Union[Literal[1], Literal[-1]]]]
NameWatchers: TypeAlias = dict[tuple[str, int], tuple[CodeStatementWithExpression, set[tuple[str, int]], list[Namespace], Optional[DreamberdPromise]]]
WhenStatementWatchers: TypeAlias = list[dict[Union[str, int], list[tuple[ExpressionTreeNode, list[tuple[CodeStatement, ...]]]]]]  # bro there are six square brackets...

def get_built_expression(expr: Union[list[Token], ExpressionTreeNode]) -> ExpressionTreeNode:
    return expr if isinstance(expr, ExpressionTreeNode) else build_expression_tree(filename, expr, code)

def get_modified_next_name(name: str, ns: int) -> str:
    return f"{name}_{ns}__next"

def get_modified_prev_name(name: str) -> str:
    return f"{name.replace('.', '__')}__prev"

# i believe this function is exclusively called from the evaluate_expression function
def evaluate_normal_function(expr: FunctionNode, func: Union[DreamberdFunction, BuiltinFunction], namespaces: list[Namespace], args: list[Value], when_statement_watchers: WhenStatementWatchers) -> Value:

    # check to evaluate builtin
    if isinstance(func, BuiltinFunction):
        if func.arg_count > len(args):
            raise_error_at_token(filename, code, f"Expected more arguments for function call with {func.arg_count} argument{'s' if func.arg_count != 1 else ''}.", expr.name)
        max_arg_count = func.arg_count if func.arg_count >= 0 else len(args)
        return func.function(*args[:max_arg_count]) or DreamberdUndefined()
    
    # check length is proper, adjust namespace, and run this code
    if len(func.args) > len(args):
        raise_error_at_token(filename, code, f"Expected more arguments for function call with {len(func.args)} argument{'s' if len(func.args) != 1 else ''}.", expr.name)
    new_namespace: Namespace = {name: Name(name, arg) for name, arg in zip(func.args, args)}
    return interpret_code_statements(func.code, namespaces + [new_namespace], [], when_statement_watchers + [{}]) or DreamberdUndefined()

def register_async_function(expr: FunctionNode, func: DreamberdFunction, namespaces: list[Namespace], args: list[Value], async_statements: AsyncStatements) -> None:
    """ Adds a job to the async statements queue, which is accessed in the interpret_code_statements function. """
    if len(func.args) > len(args):
        raise_error_at_token(filename, code, f"Expected more arguments for function call with {len(func.args)} argument{'s' if len(func.args) != 1 else ''}.", expr.name)
    function_namespaces = namespaces + [{name: Name(name, arg) for name, arg in zip(func.args, args)}]
    async_statements.append((func.code, function_namespaces, 0, 1))

def get_code_from_when_statement_watchers(name_or_id: Union[str, int], when_statement_watchers: WhenStatementWatchers) -> list[tuple[ExpressionTreeNode, list[tuple[CodeStatement, ...]]]]:
    vals = []
    for watcher_dict in when_statement_watchers:
        if val := watcher_dict.get(name_or_id):
            vals += val
    return vals

def remove_from_when_statement_watchers(name_or_id: Union[str, int], watcher: tuple[ExpressionTreeNode, list[tuple[CodeStatement, ...]]], when_statement_watchers: WhenStatementWatchers) -> None:
    for watcher_dict in when_statement_watchers:
        if vals := watcher_dict.get(name_or_id):
            remove = None
            for i, v in enumerate(vals):
                if v == watcher:
                    remove = i
            if remove is not None: 
                del vals[remove]

def remove_from_all_when_statement_watchers(name_or_id: Union[str, int], when_statement_watchers: WhenStatementWatchers) -> None: 
    for watcher_dict in when_statement_watchers:
        if name_or_id in watcher_dict:
            del watcher_dict[name_or_id]

def load_global_dreamberd_variables(namespaces: list[Namespace]) -> None:

    dir_path = Path().home()/DB_RUNTIME_PATH
    inf_values_path = dir_path/INF_VAR_VALUES_PATH
    inf_var_list = dir_path/INF_VAR_PATH
    if not dir_path.is_dir(): return 
    if not inf_values_path.is_dir(): return 
    if not inf_var_list.is_file(): return 

    with open(inf_var_list, 'r') as f:
        for line in f.readlines():
            if not line.strip():
                continue
        
            name, identity, can_be_reset, can_edit_value, confidence = line.split(DB_VAR_TO_VALUE_SEP)
            can_be_reset = eval(can_be_reset) if can_be_reset in ["True", "False"] else True # safe code !!!!!!!!!!!!
            can_edit_value = eval(can_edit_value) if can_edit_value in ["True", "False"] else True

            with open(dir_path/INF_VAR_VALUES_PATH/identity, "rb") as data_f:
                value = pickle.load(data_f)
            namespaces[-1][name] = Variable(name, [VariableLifetime(value, 100000000000, int(confidence), can_be_reset, can_edit_value)], [])

def load_public_global_variables(namespaces: list[Namespace]) -> None:
    repo_url = "https://raw.githubusercontent.com/vivaansinghvi07/dreamberd-interpreter-globals/main"
    for line in requests.get(f"{repo_url}/public_globals.txt").text.split("\n"):
        if not line.strip(): continue
        name, address, confidence = line.split(DB_VAR_TO_VALUE_SEP)
        can_be_reset = can_edit_value = False  # these were const 

        encoded_value = requests.get(f"{repo_url}/global_objects/{address}").text
        byte_list = [int(encoded_value[i:i+4]) for i in range(0, len(encoded_value), 4)]
        value = pickle.loads(bytearray(byte_list))
        namespaces[-1][name] = Variable(name, [VariableLifetime(value, 100000000000, int(confidence), can_be_reset, can_edit_value)], [])

def open_global_variable_issue(name: str, value: Value, confidence: int):
    if not GITHUB_IMPORTED:
        raise_error_at_line(filename, code, current_line, "Cannot create a public global variable without a the GitHub API imported.")
    try:
        access_token = os.environ["GITHUB_ACCESS_TOKEN"]
    except KeyError:
        raise_error_at_line(filename, code, current_line, "To declare public globals, you must set the GITHUB_ACCESS_TOKEN to a personal access token.")

    # transform the variable into a value string
    value_bytes = list(pickle.dumps(value))
    issue_body = "".join([str(b).rjust(4, '0') for b in value_bytes])

    # post the variable as an issue to the main github repo
    with github.Github(auth=github.Auth.Token(access_token)) as g:   # type: ignore 
        repo = g.get_repo("vivaansinghvi07/dreamberd-interpreter-globals")
        repo.create_issue(f"Create Public Global: {name}{DB_VAR_TO_VALUE_SEP}{confidence}", issue_body)

def declare_new_variable(statement: VariableDeclaration, value: Value, namespaces: list[Namespace], async_statements: AsyncStatements, when_statement_watchers: WhenStatementWatchers):

    name, lifetime, confidence, debug, modifiers = statement.name.value, statement.lifetime, statement.confidence, statement.debug, statement.modifiers 
    name_token = statement.name  # for error handling purposes
    is_global = len(modifiers) == 3
    can_be_reset = isinstance(v := get_value_from_namespaces(modifiers[-2], namespaces), DreamberdKeyword) and v.value == "var"
    can_edit_value = isinstance(v := get_value_from_namespaces(modifiers[-1], namespaces), DreamberdKeyword) and v.value == "var"

    if '.' in name:
        raise_error_at_token(filename, code, "Cannot declare a variable with periods in the name.", name_token)

    is_lifetime_temporal = lifetime is not None and not lifetime[-1].isdigit()
    variable_duration = 100000000000 if is_lifetime_temporal or lifetime is None else int(lifetime)
    target_lifetime = VariableLifetime(value, variable_duration, confidence, can_be_reset, can_edit_value)

    if v := namespaces[-1].get(name): 
        if isinstance(v, Variable):   # check for another declaration?
            target_var = v
            for i in range(len(v.lifetimes) + 1):
                if i == len(v.lifetimes) or v.lifetimes[i].confidence == confidence:
                    if i == 0:
                        v.prev_values.append(v.value)
                    v.lifetimes[i:i] = [target_lifetime]
        else:
            target_var = Variable(name, [target_lifetime], [v.value])
            namespaces[-1][name] = target_var
    else:  # for loop finished unbroken, no matches found
        target_var = Variable(name, [target_lifetime], [])
        namespaces[-1][name] = target_var

    match debug:
        case 0: pass 
        case 1: 
            debug_print(filename, code, f"Setting {statement.name.value} to {db_to_string(value).value}", statement.name)
        case 2: 
            expr = get_built_expression(statement.expression)
            debug_print(filename, code, f"Setting {' '.join([mod.value for mod in statement.modifiers])} variable \"{statement.name.value}\" to {db_to_string(value).value} with a lifetime of {lifetime}.", statement.name)
        case 3: 
            expr = get_built_expression(statement.expression)
            names = gather_names_or_values(expr)
            debug_print(filename, code, f"Setting {' '.join([mod.value for mod in statement.modifiers])} variable \"{statement.name.value}\" to {db_to_string(value).value} with a lifetime of {lifetime}.\nThe value of each name in the expression is the following: \n{chr(10).join([f'  {name}: {db_to_string(get_value_from_namespaces(name_token, namespaces)).value}' for name in names])}", statement.name)
        case _: 
            expr = get_built_expression(statement.expression)
            names = gather_names_or_values(expr)
            debug_print(filename, code, f"Setting {' '.join([mod.value for mod in statement.modifiers])} variable \"{statement.name.value}\" to {db_to_string(value).value} with a lifetime of {lifetime}.\nThe value of each name in the exprehttps://accounts.spotify.com/en/login?continue=https%3A%2F%2Fopen.spotify.com%2Fplaylist%2F0jqXRIN34QAfwEiB72S4H7%3Fsi%3D6f53123059f741ec%26pt%3D09c9c4118ece8f2d0fad56d9f2a4345ession is the following: \n{chr(10).join([f'  {name}: {db_to_string(get_value_from_namespaces(name_token, namespaces)).value}' for name in names])}\nThe expression used to get this value is: \n{expr.to_string()}", statement.name)

    # check if there is a watcher for this name
    watchers_key = (name, id(namespaces[-1]))
    if watcher := name_watchers.get(watchers_key):
        st, stored_nexts, watcher_ns, promise = watcher
        mod_name = get_modified_next_name(*watchers_key)
        watcher_ns[-1][mod_name] = Name(mod_name, value)  # add the value to the uppermost namespace
        stored_nexts.remove(watchers_key)                   # remove the name from the set containing remaining names
        if not stored_nexts:  # not waiting on anybody else, execute the code
            interpret_name_watching_statement(st, watcher_ns, promise, async_statements, when_statement_watchers)
        del name_watchers[watchers_key]             # stop watching this name

    # check if this name appears in a when statement of the appropriate scope  --  it would have to be watching the name
    if when_watchers := get_code_from_when_statement_watchers(name, when_statement_watchers):
        for when_watcher in when_watchers:  # i just wanna be done with this :(
            condition, inside_statements = when_watcher
            condition_val = evaluate_expression(condition, namespaces, async_statements, when_statement_watchers)
            if isinstance(value, DreamberdMutable):
                when_statement_watchers[-1][id(value)].append(when_watcher)  ##### remember : this is tuple so it is immutable and copied !!!!!!!!!!!!!!!!!!!!!!  # wait nvm i suick at pytghon
            if isinstance(target_var.prev_values[-1], DreamberdMutable):   # if prev value was being observed under this statement, remove it  ??
                remove_from_when_statement_watchers(id(target_var.prev_values[-1]), when_watcher, when_statement_watchers)
            when_statement_watchers[-1][id(target_var)].append(when_watcher)  # put this where the new variable is
            execute_conditional(condition_val, inside_statements, namespaces, when_statement_watchers)
        remove_from_all_when_statement_watchers(name, when_statement_watchers)  # that name is now set to a variable, discard it from the when statement  --  it is now a var not a string

    if is_global:
        open_global_variable_issue(name, value, confidence)

    # if we're dealing with seconds just sleep in another thread and remove the variable lifetime
    if lifetime == "Infinity":
        # if len(namespaces) == 1: continue  # only save global vars if they are in the global scope

        # define and initialize the directories
        dir_path = Path().home()/DB_RUNTIME_PATH
        inf_values_path = dir_path/INF_VAR_VALUES_PATH
        if not dir_path.is_dir(): dir_path.mkdir()
        if not inf_values_path.is_dir(): inf_values_path.mkdir()

        generated_addr = random.randint(1, 100000000000)  # hopefully never repeat, if it does, oh well :)
        with open(dir_path/INF_VAR_PATH, 'a') as f:
            SEP = DB_VAR_TO_VALUE_SEP
            f.write(f"{name}{SEP}{generated_addr}{SEP}{can_be_reset}{SEP}{can_edit_value}{SEP}{confidence}\n")
        with open(dir_path/INF_VAR_VALUES_PATH/str(generated_addr), "wb") as f:
            pickle.dump(value, f)
            
    elif is_lifetime_temporal:
        def remove_lifetime(lifetime: str, target_var: Variable, target_lifetime: VariableLifetime, error_line: int):
            if lifetime[-1] not in ['s', 'm'] or all(c.isdigit() for c in lifetime[:-1]):
                raise_error_at_line(filename, code, error_line, "Invalid time unit for variable lifetime.")
            sleep(int(lifetime[:-1]) if lifetime[-1] == 's' else int(lifetime[:-1] * 60))
            for i, lt in reversed([*enumerate(target_var.lifetimes)]):
                if lt is target_lifetime:
                    del target_var.lifetimes[i]
        Thread(target=remove_lifetime, args=(lifetime, target_var, target_lifetime)).start()

def assign_variable(statement: VariableAssignment, indexes: list[Value], new_value: Value, namespaces: list[Namespace], async_statements: AsyncStatements, when_statement_watchers: WhenStatementWatchers):
    name, confidence, debug = statement.name.value, statement.confidence, statement.debug
    name_token = statement.name
        
    var, ns = get_name_and_namespace_from_namespaces(name, namespaces)
    if var is None:
        raise_error_at_token(filename, code, "Attempted to set a name that is undefined.", name_token)
 
    match debug:
        case 0: pass 
        case 1: 
            debug_print(filename, code, f"Setting {statement.name.value}{''.join([f'[{db_to_string(val).value}]' for val in indexes])} to {db_to_string(new_value).value}", statement.name)
        case 2:
            expr = get_built_expression(statement.expression)
            names = gather_names_or_values(expr)
            debug_print(filename, code, f"Setting {statement.name.value}{''.join([f'[{db_to_string(val).value}]' for val in indexes])} to {db_to_string(new_value).value}\nThe value of each name in the expression is the following: \n{chr(10).join([f'  {name}: {db_to_string(get_value_from_namespaces(name, namespaces)).value}' for name in names])}", statement.name)
        case 3: 
            expr = get_built_expression(statement.expression)
            names = gather_names_or_values(expr)
            debug_print(filename, code, f"Setting {statement.name.value}{''.join([f'[{db_to_string(val).value}]' for val in indexes])} to {db_to_string(new_value).value}\nThe value of each name in the expression is the following: \n{chr(10).join([f'  {name}: {db_to_string(get_value_from_namespaces(name, namespaces)).value}' for name in names])}\nThe expression used to get this value is: \n{expr.to_string()}", statement.name)
        case _:
            expr = get_built_expression(statement.expression)
            index_exprs = [get_built_expression(ex) for ex in statement.indexes]
            names = gather_names_or_values(expr)
            for ex in index_exprs:
                names |= gather_names_or_values(ex)
            debug_print(filename, code, f"Setting {statement.name.value}{''.join([f'[{db_to_string(val).value}]' for val in indexes])} to {db_to_string(new_value).value}\nThe value of each name in the program is the following: \n{chr(10).join([f'  {name}: {db_to_string(get_value_from_namespaces(name, namespaces)).value}' for name in names])}\nThe expression used to get this value is: \n{expr.to_string()}\nThe expression used to get the indexes are as follows: \n{(chr(10) * 2).join([ex.to_string(1) for ex in index_exprs])}", statement.name)

    visited_whens = []
    if indexes:

        # goes down the list until it can assign something in the list
        def assign_variable_helper(value_to_modify: Value, remaining_indexes: list[Value]):
            if not value_to_modify or not isinstance(value_to_modify, DreamberdIndexable):
                raise_error_at_line(filename, code, name_token.line, "Attempted to index into an un-indexable object.")
            index = remaining_indexes.pop(0) 

            # check for some watchers here too!!!!!!!!!!!
            when_watchers = get_code_from_when_statement_watchers(id(value_to_modify), when_statement_watchers)
            for when_watcher in when_watchers:  # i just wanna be done with this :(
                if any([when_watcher == x for x in visited_whens]): 
                    continue
                condition, inside_statements = when_watcher
                condition_val = evaluate_expression(condition, namespaces, async_statements, when_statement_watchers)
                execute_conditional(condition_val, inside_statements, namespaces, when_statement_watchers)
                visited_whens.append(when_watcher)

            if not remaining_indexes:  # perform actual assignment here
                value_to_modify.assign_index(index, new_value)
            else:
                assign_variable_helper(value_to_modify.access_index(index), remaining_indexes)

        if isinstance(var, Variable) and not var.can_edit_value:
            raise_error_at_token(filename, code, "Cannot edit the value of this variable.", name_token)
        assign_variable_helper(var.value, indexes)
               
    else: 
        if not isinstance(var, Variable):
            raise_error_at_token(filename, code, "Attempted to set name that is not a variable.", name_token)
        if not var.can_be_reset:
            raise_error_at_token(filename, code, "Attempted to set a variable that cannot be set.", name_token)
        var.add_lifetime(new_value, confidence, 100000000000, var.can_be_reset, var.can_edit_value)

    # check if there is anything watching this value
    watchers_key = (name.split('.')[-1], id(ns))  # this shit should be a seperate function
    if watcher := name_watchers.get(watchers_key):
        st, stored_nexts, watcher_ns, promise = watcher
        mod_name = get_modified_next_name(*watchers_key)
        watcher_ns[-1][mod_name] = Name(mod_name, new_value)  # add the value to the uppermost namespace
        stored_nexts.remove(watchers_key)                   # remove the name from the set containing remaining names
        if not stored_nexts:  # not waiting on anybody else, execute the code
            interpret_name_watching_statement(st, watcher_ns, promise, async_statements, when_statement_watchers)
        del name_watchers[watchers_key]             # stop watching this name  

    # get new watchers for this
    when_watchers = get_code_from_when_statement_watchers(id(var), when_statement_watchers)
    for when_watcher in when_watchers:  # i just wanna be done with this :(
        if any([when_watcher == x for x in visited_whens]): 
            continue
        condition, inside_statements = when_watcher
        condition_val = evaluate_expression(condition, namespaces, async_statements, when_statement_watchers)
        if isinstance(new_value, DreamberdMutable):
            if id(new_value) not in when_statement_watchers[-1]:
                when_statement_watchers[-1][id(new_value)] = []  
            when_statement_watchers[-1][id(new_value)].append(when_watcher)  ##### remember : this is tuple so it is immutable and copied !!!!!!!!!!!!!!!!!!!!!!  # wait nvm i suick at pytghon
        if isinstance(var, Variable) and var.prev_values and isinstance(var.prev_values[-1], DreamberdMutable):   # if prev value was being observed under this statement, remove it  ??
            remove_from_when_statement_watchers(id(var.prev_values[-1]), when_watcher, when_statement_watchers)
        execute_conditional(condition_val, inside_statements, namespaces, when_statement_watchers)
        visited_whens.append(when_watcher)

def get_value_from_promise(val: DreamberdPromise) -> Value:
    if val.value is None:
        return DreamberdUndefined()
    return val.value

def get_name_from_namespaces(name: str, namespaces: list[Namespace]) -> Optional[Union[Variable, Name]]:
    """ This is called when we are sure that the value is a name. """
    if len(name_split := name.split('.')) == 1:
        for ns in reversed(namespaces):
            if (v := ns.get(name)) is not None:
                return v
    else:
        base_val = get_name_from_namespaces(name_split[0], namespaces)
        if not base_val:   # base object not found
            return None
        for other_name in name_split[1:]:
            if not isinstance(base_val.value, DreamberdNamespaceable):
                return None
            base_val = get_name_from_namespaces(other_name, [base_val.value.namespace])
            if not base_val:   # the value was not found in the namespace
                return None
        return base_val
    return None

def get_name_and_namespace_from_namespaces(name: str, namespaces: list[Namespace]) -> tuple[Optional[Union[Variable, Name]], Optional[Namespace]]:
    """ This is the same as the function defined above, except it also returns the namespace in which the name was found. """
    if len(name_split := name.split('.')) == 1:
        for ns in reversed(namespaces):
            if (v := ns.get(name)) is not None:
                return v, ns
    else:
        base_val, ns = get_name_and_namespace_from_namespaces(name_split[0], namespaces)
        if not base_val:
            return None, ns  # value doesn't exist but the namespace does
        for other_name in name_split[1:]:
            if not isinstance(base_val.value, DreamberdNamespaceable):
                return None, None 
            ns = base_val.value.namespace
            base_val = get_name_from_namespaces(other_name, [ns])
            if not base_val:  # same case as before
                return None, ns
        return base_val, ns
    return None, namespaces[-1]

def evaluate_escape_sequences(val: DreamberdString) -> DreamberdString:  # this needs only be called once per completed string
    return DreamberdString(eval(f'"{val.value.replace(f"{chr(34)}", f"{chr(92)}{chr(34)}")}"'))  # cursed string parsing

def interpret_formatted_string(val: Token, namespaces: list[Namespace], async_statements: AsyncStatements, when_statement_watchers: WhenStatementWatchers) -> DreamberdString:
    val_string = val.value
    locale.setlocale(locale.LC_ALL, locale.getlocale()[0])
    symbol: str = locale.localeconv()['currency_symbol']  # type: ignore
    if not any(indeces := [val_string[i:i+len(symbol)] == symbol for i in range(len(val_string) - len(symbol))]):
        return DreamberdString(val_string)
    try:
        evaluated_values: list[tuple[str, tuple[int, int]]] = []  # [(str, (start, end))...]
        for group_start_index in [i for i in range(len(indeces)) if indeces[i]]:
            if val_string[group_start_index + len(symbol)] == '{':
                end_index = group_start_index + len(symbol)
                bracket_layers = 1
                while bracket_layers:  # if this errs, it will be caught and detected as invalid formatting
                    end_index += 1
                    if val_string[end_index] == '{':
                        bracket_layers += 1 
                    elif val_string[end_index] == '}':
                        bracket_layers -= 1
                
                # end_index is now the index containing the bracket.
                internal_tokens = db_tokenize(f"{filename}__interpolated_string", val_string[group_start_index + len(symbol) + 1 : end_index ])
                internal_expr = build_expression_tree(filename, internal_tokens, code)
                internal_value = evaluate_expression(internal_expr, namespaces, async_statements, when_statement_watchers, ignore_string_escape_sequences=True)
                evaluated_values.append((db_to_string(internal_value).value, (group_start_index, end_index + 1)))

        new_string = list(val_string)
        for replacement, (start, end) in reversed(evaluated_values):
            new_string[start:end] = replacement
        return DreamberdString(''.join(new_string))
                
    except IndexError:
        raise_error_at_line(filename, code, current_line, "Invalid interpolated string formatting.")

def determine_non_name_value(val: Token) -> Value:
    """ 
    Takes a string/Token and determines if the value is a number, string, or invalid. 
    Valid names should have been found already by the previous function.
    """
    global deleted_values
    retval = None
    if len(v := val.value.split('.')) <= 2:
        if all(x.isdigit() for x in v):
            retval = DreamberdNumber([int, float][len(v) - 1](val.value))
    if not retval:
        retval = DreamberdString(val.value)
    if retval in deleted_values:
        raise_error_at_line(filename, code, val.line, f"The value {retval.value} has been deleted.")
    return retval

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
            return DreamberdBoolean(left_num == right_num or (False if left_num == 0 else (left_num - right_num) / left_num < NUM_EQUALITY_RATIO))

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
    match left, right:  # i know these are horribly verbose but if i don't do this my LSP yells at me
        case (DreamberdNumber(), DreamberdNumber()) | \
             (DreamberdString(), DreamberdString()) | \
             (DreamberdBoolean(), DreamberdBoolean()) | \
             (DreamberdKeyword(), DreamberdKeyword()): 
            return DreamberdBoolean(left.value == right.value)
        case (DreamberdUndefined(), DreamberdUndefined()):
            return DreamberdBoolean(True) 
        case (DreamberdObject(), DreamberdObject()):
            return DreamberdBoolean(left.class_name == right.class_name and 
                                    left.namespace.keys() == right.namespace.keys() and 
                                    all([is_really_equal(left.namespace[k].value, right.namespace[k].value).value for k in left.namespace.keys()]))
        case (DreamberdFunction(), DreamberdFunction()):
            return DreamberdBoolean(all([getattr(left, name) == getattr(right, name) for name in ["code", "args", "is_async"]]))
        case (DreamberdList(), DreamberdList()):
            return DreamberdBoolean(len(left.values) == len(right.values) and 
                                    all([is_really_equal(l, r).value for l, r in zip(left.values, right.values)]))
        case (DreamberdMap(), DreamberdMap()):
            return DreamberdBoolean(left.self_dict.keys() == right.self_dict.keys() and 
                                    all([is_really_equal(left.self_dict[k], right.self_dict[k]).value for k in left.self_dict]))
    return DreamberdBoolean(None)

def is_really_really_equal(left: Value, right: Value) -> DreamberdBoolean:
    return DreamberdBoolean(left is right)

def is_less_than(left: Value, right: Value) -> DreamberdBoolean:
    if type(left) != type(right):
        raise_error_at_line(filename, code, current_line, f"Cannot compare value of type {type(left).__name__} with one of type {type(right).__name__}.")
    match left, right:
        case (DreamberdNumber(), DreamberdNumber()) | \
             (DreamberdString(), DreamberdString()) | \
             (DreamberdBoolean(), DreamberdBoolean()):
            if isinstance(left, DreamberdBoolean) and isinstance(right, DreamberdBoolean) and (left.value is None or right.value is None):
                return DreamberdBoolean(None)
            return DreamberdBoolean(left.value < right.value)   # type: ignore
        case (DreamberdUndefined(), DreamberdUndefined()):
            return DreamberdBoolean(False)
        case (DreamberdList(), DreamberdList()):
            return DreamberdBoolean(len(left.values) < len(right.values))
        case (DreamberdMap(), DreamberdMap()):
            return DreamberdBoolean(len(left.self_dict) < len(right.self_dict))
        case (DreamberdKeyword(), DreamberdKeyword()) | \
             (DreamberdObject(), DreamberdObject()) | \
             (DreamberdFunction(), DreamberdFunction()):
            raise_error_at_line(filename, code, current_line, f"Comparison not supported between elements of type {type(left).__name__}.")
    return DreamberdBoolean(None)

def perform_single_value_operation(val: Value, operator_token: Token) -> Value: 
    match operator_token.type:
        case TokenType.SUBTRACT:
            match val:
                case DreamberdNumber():
                    return DreamberdNumber(-val.value)
                case DreamberdList():
                    return DreamberdList(val.values[::-1])
                case DreamberdString():
                    return DreamberdString(val.value[::-1])
                case _:
                    raise_error_at_token(filename, code, f"Cannot negate a value of type {type(val).__name__}", operator_token)
        case TokenType.SEMICOLON:
            val_bool = db_to_boolean(val)
            return db_not(val_bool) 
    raise_error_at_token(filename, code, "Something went wrong. My bad.", operator_token)

def perform_two_value_operation(left: Value, right: Value, operator: OperatorType, operator_token: Token) -> Value:
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
                raise_error_at_line(filename, code, current_line, "Cannot raise a negative base to a non-integer exponent.")
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
        case OperatorType.GT | OperatorType.LE:
            is_eq = is_really_equal(left, right)
            is_less = is_less_than(left, right)
            is_le = False
            match is_eq.value, is_less.value:  # performs the OR operation
                case (True, _) | (_, True): is_le = True
                case (None, _) | (_, None): is_le = None 
            if operator == OperatorType.LE:
                return DreamberdBoolean(is_le)
            return db_not(DreamberdBoolean(is_le))
        case OperatorType.LT | OperatorType.GE:  
            if operator == OperatorType.LT:
                return is_less_than(left, right)
            return db_not(is_less_than(left, right))

    raise_error_at_token(filename, code, "Something went wrong here.", operator_token)

def get_value_from_namespaces(name_or_value: Token, namespaces: list[Namespace]) -> Value:

    # what the frick am i doing rn
    if v := get_name_from_namespaces(name_or_value.value, namespaces):
        if isinstance(v.value, DreamberdPromise):
            return deepcopy(get_value_from_promise(v.value))  # consider not deepcopying this but it doesnt really matter
        return v.value
    return determine_non_name_value(name_or_value)

def print_expression_debug(debug: int, expr: Union[list[Token], ExpressionTreeNode], value: Value, namespaces: list[Namespace]) -> None:
    expr = get_built_expression(expr)
    msg = None
    match debug:
        case 0: pass 
        case 1: 
            msg = f"Expression evaluates to value {db_to_string(value).value}."
        case 2: 
            names = gather_names_or_values(expr)
            msg = f"Expression evaluates to value {db_to_string(value).value}.\nThe value of each name in the expression is the following: \n{chr(10).join([f'  {name}: {db_to_string(get_value_from_namespaces(name, namespaces)).value}' for name in names])}"
        case _: 
            names = gather_names_or_values(expr)
            msg = f"Expression evaluates to value {db_to_string(value).value}.\nThe value of each name in the expression is the following: \n{chr(10).join([f'  {name}: {db_to_string(get_value_from_namespaces(name, namespaces)).value}' for name in names])}\nThe expression used to get this value is: \n{expr.to_string()}"

    if not msg: return
    if t := get_expr_first_token(expr):
        debug_print(filename, code, msg, t)
    else: debug_print_no_token(filename, msg)


def evaluate_expression(expr: Union[list[Token], ExpressionTreeNode], namespaces: list[dict[str, Union[Variable, Name]]], async_statements: AsyncStatements, when_statement_watchers: WhenStatementWatchers, *, ignore_string_escape_sequences: bool = False) -> Value:
    """ Wrapper for the evaluate_expression_for_real function that checks deleted values on each run. """
    retval = evaluate_expression_for_real(expr, namespaces, async_statements, when_statement_watchers, ignore_string_escape_sequences)
    if isinstance(retval, (DreamberdNumber, DreamberdString)) and retval in deleted_values:
        raise_error_at_line(filename, code, current_line, f"The value {retval.value} has been deleted.")
    return retval

def evaluate_expression_for_real(expr: Union[list[Token], ExpressionTreeNode], namespaces: list[dict[str, Union[Variable, Name]]], async_statements: AsyncStatements, when_statement_watchers: WhenStatementWatchers, ignore_string_escape_sequences: bool) -> Value:

    expr = get_built_expression(expr)
    match expr:
        case FunctionNode():  # done :)
            
            # for a function, the thing must be in the namespace
            func = get_name_from_namespaces(expr.name.value, namespaces)

            # make sure it exists and it is actually a function in the namespace
            if func is None:
                raise_error_at_token(filename, code, "Cannot find token in namespace.", expr.name)
    
            # check the thing in the await symbol. if awaiting a single function that is async, evaluate it as not async
            force_execute_sync = False
            if isinstance(func.value, DreamberdKeyword):
                if func.value.value == "await":
                    if len(expr.args) != 1:
                        raise_error_at_token(filename, code, "Expected only one argument for await function.", expr.name)
                    if not isinstance(expr.args[0], FunctionNode):
                        raise_error_at_token(filename, code, "Expected argument of await function to be a function call.", expr.name)
                    force_execute_sync = True 
                    
                    # check for None again
                    expr = expr.args[0]
                    func = get_name_from_namespaces(expr.name.value, namespaces)
                    if func is None:  # the other check happens in the next statement
                        raise_error_at_token(filename, code, "Cannot find token in namespaces.", expr.name)

                elif func.value.value == "previous":
                    if len(expr.args) != 1:
                        raise_error_at_token(filename, code, "Expected only one argument for previous function.", expr.name)
                    if not isinstance(expr.args[0], ValueNode):
                        raise_error_at_token(filename, code, "Expected argument of previous function to be a variable.", expr.name)
                    force_execute_sync = True 
                    
                    # check for None again
                    val = get_name_from_namespaces(expr.args[0].name_or_value.value, namespaces)
                    if not isinstance(val, Variable):
                        raise_error_at_token(filename, code, "Expected argument of previous function to be a defined variable.", expr.args[0].name_or_value)
                    return val.prev_values[-1]

            if not isinstance(func.value, (BuiltinFunction, DreamberdFunction)):
                raise_error_at_token(filename, code, "Attempted function call on non-function value.", expr.name)
            
            caller = None
            if len(name_split := expr.name.value.split('.')) > 1:
                caller = '.'.join(name_split[:-1])
                expr = deepcopy(expr)   # we create a copy of the expression as to not modify it badly
                expr.args.insert(0, ValueNode(Token(TokenType.NAME, caller, expr.name.line, expr.name.col)))  # artificially put this here, as this is the imaginary "this" 
            args = [evaluate_expression(arg, namespaces, async_statements, when_statement_watchers) for arg in expr.args]
            if isinstance(args[0], DreamberdSpecialBlankValue):
                args = args[1:]
            if isinstance(func.value, DreamberdFunction) and func.value.is_async and not force_execute_sync:
                register_async_function(expr, func.value, namespaces, args, async_statements)
                return DreamberdUndefined()
            elif isinstance(func.value, BuiltinFunction) and func.value.modifies_caller:  # special cases where the function itself modifies the caller
                if caller:  # seems like a needless check but it makes the errors go away
                    caller_var = get_name_from_namespaces(caller, namespaces)
                    if isinstance(caller_var, Variable) and not caller_var.can_edit_value:
                        raise_error_at_line(filename, code, current_line, "Cannot edit the value of this variable.")

                retval = evaluate_normal_function(expr, func.value, namespaces, args, when_statement_watchers)
                when_watchers = get_code_from_when_statement_watchers(id(args[0]), when_statement_watchers)
                for when_watcher in when_watchers:  # i just wanna be done with this :(
                    condition, inside_statements = when_watcher
                    condition_val = evaluate_expression(condition, namespaces, async_statements, when_statement_watchers)
                    execute_conditional(condition_val, inside_statements, namespaces, when_statement_watchers)
                return retval

            return evaluate_normal_function(expr, func.value, namespaces, args, when_statement_watchers)

        case ListNode():  # done :) 
            return DreamberdList([evaluate_expression(x, namespaces, async_statements, when_statement_watchers) for x in expr.values])

        case ValueNode():  # done :)
            if expr.name_or_value.type == TokenType.STRING: 
                retval = interpret_formatted_string(expr.name_or_value, namespaces, async_statements, when_statement_watchers)
                if not ignore_string_escape_sequences:
                    return evaluate_escape_sequences(retval)
                return retval
            return get_value_from_namespaces(expr.name_or_value, namespaces)

        case IndexNode():  # done :)
            value = evaluate_expression(expr.value, namespaces, async_statements, when_statement_watchers)
            index = evaluate_expression(expr.index, namespaces, async_statements, when_statement_watchers)
            if not isinstance(value, DreamberdIndexable):
                raise_error_at_line(filename, code, current_line, "Attempting to index a value that is not indexable.")
            return value.access_index(index)

        case ExpressionNode():  # done :)
            left = evaluate_expression(expr.left, namespaces, async_statements, when_statement_watchers)
            if db_to_boolean(left).value == True and expr.operator == OperatorType.OR:   # handle short curcuiting for True or __
                return left
            elif db_to_boolean(left).value == False and expr.operator == OperatorType.AND:   # handle short curcuiting for False and __
                return left
            right = evaluate_expression(expr.right, namespaces, async_statements, when_statement_watchers)
            return perform_two_value_operation(left, right, expr.operator, expr.operator_token)

        case SingleOperatorNode():
            val = evaluate_expression(expr.expression, namespaces, async_statements, when_statement_watchers)
            return perform_single_value_operation(val, expr.operator)

    return DreamberdUndefined()

def handle_next_expressions(expr: ExpressionTreeNode, namespaces: list[Namespace]) -> tuple[ExpressionTreeNode, set[tuple[str, int]], set[str]]:

    """ 
    This function looks for the "next" keyword in an expression, and detects seperate await modifiers for that keyword.
    Then, it removes the "next" and "await next" nodes from the ExpressionTree, and returns the head of the tree.
    Additionally, every name that appears in the function as a next or async next, its value is saved in a temporary namespace.
    With the returned set of names that are used in "next" and "await next", we can insert these into a dictionary
        that contains information about which names are being "watched" for changes. When this dictionary changes,
        we can execute code accordingly.

    This is my least favorite function in the program.
    """

    normal_nexts: set[tuple[str, int]] = set()
    async_nexts: set[str] = set()
    inner_nexts: list[tuple[set[tuple[str, int]], set[str]]] = []
    match expr:
        case FunctionNode():

            func = get_name_from_namespaces(expr.name.value, namespaces)
            if func is None:
                raise_error_at_token(filename, code, "Attempted function call on undefined variable.", expr.name)

            # check if it is a next or await 
            is_next = is_await = False   # i don't need this but it makes my LSP stop crying so it's here
            if isinstance(func.value, DreamberdKeyword) and \
               ((is_next := func.value.value == "next") or (is_await := func.value.value == "await")):

                if is_next:

                    # add it to list of things to watch for and change the returned expression to the name being next-ed
                    if len(expr.args) != 1 or not isinstance(expr.args[0], ValueNode):
                        raise_error_at_token(filename, code, "\"Next\"keyword can only take a single value as an argument.", expr.name)
                    name = expr.args[0].name_or_value.value
                    _, ns = get_name_and_namespace_from_namespaces(name, namespaces)
                    if not ns:
                        raise_error_at_line(filename, code, current_line, "Attempted to access namespace of a value without a namespace.")
                    last_name = name.split('.')[-1]
                    normal_nexts.add((name, id(ns)))
                    expr = expr.args[0]
                    expr.name_or_value.value = get_modified_next_name(last_name, id(ns))

                elif is_await:

                    if len(expr.args) != 1 or not isinstance(expr.args[0], FunctionNode):
                        raise_error_at_token(filename, code, "Can only await a function.", expr.name)
                    inner_expr = expr.args[0]
                        
                    func = get_name_from_namespaces(expr.args[0].name.value, namespaces)
                    if func is None:
                        raise_error_at_token(filename, code, "Attempted function call on undefined variable.", expr.name)

                    if isinstance(func.value, DreamberdKeyword) and func.value.value == "next":
                        if len(inner_expr.args) != 1 or not isinstance(inner_expr.args[0], ValueNode):
                            raise_error_at_token(filename, code, "\"Next\"keyword can only take a single value as an argument.", inner_expr.name)
                        name = inner_expr.args[0].name_or_value.value 
                        _, ns = get_name_and_namespace_from_namespaces(name, namespaces)
                        if not ns:
                            raise_error_at_line(filename, code, current_line, "Attempted to access namespace of a value without a namespace.")
                        last_name = name.split('.')[-1]
                        async_nexts.add(name)  # only need to store the name for the async ones because we are going to wait anyways
                        expr = inner_expr.args[0]
                        expr.name_or_value.value = get_modified_next_name(last_name, id(ns))

            else:
                replacement_args = []
                for arg in expr.args:
                    new_expr, normal_arg_nexts, async_arg_nexts = handle_next_expressions(arg, namespaces)
                    inner_nexts.append((normal_arg_nexts, async_arg_nexts))
                    replacement_args.append(new_expr)
                expr.args = replacement_args
                
        case ListNode():
            replacement_values = []
            for ex in expr.values:
                new_expr, normal_expr_nexts, async_expr_nexts = handle_next_expressions(ex, namespaces)
                inner_nexts.append((normal_expr_nexts, async_expr_nexts))
                replacement_values.append(new_expr)
            expr.values = replacement_values
        case IndexNode():
            new_value, normal_value_nexts, async_value_nexts = handle_next_expressions(expr.value, namespaces)
            new_index, normal_index_nexts, async_index_nexts = handle_next_expressions(expr.index, namespaces)
            expr.value = new_value 
            expr.index = new_index
            inner_nexts.extend([(normal_value_nexts, async_value_nexts), (normal_index_nexts, async_index_nexts)])
        case ExpressionNode():
            new_left, normal_left_nexts, async_left_nexts = handle_next_expressions(expr.left, namespaces)
            new_right, normal_right_nexts, async_right_nexts = handle_next_expressions(expr.right, namespaces)
            expr.left = new_left 
            expr.right = new_right
            inner_nexts.extend([(normal_left_nexts, async_left_nexts), (normal_right_nexts, async_right_nexts)])
        case SingleOperatorNode():
            new_expr, normal_expr_nexts, async_expr_nexts = handle_next_expressions(expr.expression, namespaces)
            expr.expression = new_expr
            inner_nexts.append((normal_expr_nexts, async_expr_nexts))
    for nn, an in inner_nexts:
        normal_nexts |= nn 
        async_nexts |= an 
    return expr, normal_nexts, async_nexts

def save_previous_values_next_expr(expr_to_modify: ExpressionTreeNode, nexts: set[str], namespaces: list[Namespace]) -> Namespace:

    saved_namespace: Namespace = {}
    match expr_to_modify:
        case ValueNode():
            if expr_to_modify.name_or_value.type == TokenType.STRING:
                return {}
            name = expr_to_modify.name_or_value.value
            if name not in nexts:
                return {}
            val = get_name_from_namespaces(name, namespaces)
            if not val:
                val = Name("", determine_non_name_value(expr_to_modify.name_or_value))
            mod_name = get_modified_prev_name(name)
            expr_to_modify.name_or_value.value = mod_name
            return {mod_name: Name(mod_name, val.value)}
        case ExpressionNode():
            left_ns = save_previous_values_next_expr(expr_to_modify.left, nexts, namespaces)
            right_ns = save_previous_values_next_expr(expr_to_modify.right, nexts, namespaces)
            return left_ns | right_ns
        case IndexNode():
            value_ns = save_previous_values_next_expr(expr_to_modify.value, nexts, namespaces)
            index_ns = save_previous_values_next_expr(expr_to_modify.index, nexts, namespaces)
            return value_ns | index_ns
        case ListNode():
            for ex in expr_to_modify.values:
                saved_namespace |= save_previous_values_next_expr(ex, nexts, namespaces)
            return saved_namespace
        case FunctionNode():
            for arg in expr_to_modify.args:
                saved_namespace |= save_previous_values_next_expr(arg, nexts, namespaces)
            return saved_namespace
        case SingleOperatorNode():
            return save_previous_values_next_expr(expr_to_modify.expression, nexts, namespaces)
    return saved_namespace

def determine_statement_type(possible_statements: tuple[CodeStatement, ...], namespaces: list[Namespace]) -> Optional[CodeStatement]:
    instance_to_keywords: dict[type[CodeStatementKeywordable], set[str]] = {
        Conditional: {'if'},
        WhenStatement: {'when'},
        AfterStatement: {'after'},
        ClassDeclaration: {'class', 'className'},
        DeleteStatement: {'delete'},
        ReverseStatement: {'reverse'},
        ImportStatement: {'import'}
    }

    for st in possible_statements:
        if isinstance(st, CodeStatementKeywordable):
            val = get_name_from_namespaces(st.keyword.value, namespaces)
            if val is not None and isinstance(val.value, DreamberdKeyword) and val.value.value in instance_to_keywords[type(st)]:
                return st
        elif isinstance(st, ReturnStatement):
            if st.keyword is None:
                return st 
            val = get_name_from_namespaces(st.keyword.value, namespaces)
            if val and isinstance(val.value, DreamberdKeyword) and val.value.value == "return":
                return st 
        elif isinstance(st, FunctionDefinition):  # allow for async and normal function definitions
            if len(st.keywords) == 1:
                val = get_name_from_namespaces(st.keywords[0].value, namespaces)
                if val and isinstance(val.value, DreamberdKeyword) and re.match(r"^f?u?n?c?t?i?o?n?$", val.value.value):
                    return st
            elif len(st.keywords) == 2:
                val = get_name_from_namespaces(st.keywords[0].value, namespaces)
                other_val = get_name_from_namespaces(st.keywords[1].value, namespaces)
                if val and other_val and isinstance(val.value, DreamberdKeyword) and isinstance(other_val.value, DreamberdKeyword) \
                   and re.match(r"^f?u?n?c?t?i?o?n?$", other_val.value.value) and val.value.value == 'async':
                    return st
        elif isinstance(st, VariableDeclaration):  # allow for const const const and normal declarations
            if len(st.modifiers) == 2:
                if all([(val := get_name_from_namespaces(mod.value, namespaces)) is not None and 
                    isinstance(val.value, DreamberdKeyword) and val.value.value in {'const', 'var'}
                    for mod in st.modifiers]):
                    return st
            elif len(st.modifiers) == 3:
                if all([(val := get_name_from_namespaces(mod.value, namespaces)) is not None and 
                    isinstance(val.value, DreamberdKeyword) and val.value.value == 'const' 
                    for mod in st.modifiers]):
                    return st
        elif isinstance(st, ExportStatement):
            if isinstance(v := get_value_from_namespaces(st.to_keyword, namespaces), DreamberdKeyword) and v.value == 'to' and \
               isinstance(v := get_value_from_namespaces(st.export_keyword, namespaces), DreamberdKeyword) and v.value == 'export':
                return st

    # now is left: expression evalulation and variable assignment
    for st in possible_statements:
        if isinstance(st, VariableAssignment):
            return st 
    for st in possible_statements:
        if isinstance(st, ExpressionStatement):
            return st
    return None 

def adjust_for_normal_nexts(statement: CodeStatementWithExpression, async_nexts: set[str], normal_nexts: set[tuple[str, int]], promise: Optional[DreamberdPromise], namespaces: list[Namespace], prev_namespace: Namespace):

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
            raise_error_at_line(filename, code, current_line, "Something went wrong with accessing the next value of a variable.")
        mod_name = get_modified_next_name(name, id(ns))
        match old_len:
            case None: new_namespace[mod_name] = Name(mod_name, v.value if isinstance(v, Name) else v.prev_values[0])
            case i:
                if not isinstance(v, Variable):
                    raise_error_at_line(filename, code, current_line, "Something went wrong.")
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
                    raise_error_at_line(filename, code, current_line, "Something went wrong.")
                new_namespace[mod_name] = Name(mod_name, v.prev_values[i])

    # the remaining values are still waiting on a result, add these to the list of name watchers
    # this new_namespace i am adding is purely for use in evaluation of expressions, and the code within 
    # a code statement should not use that namespace. therefore, some logic must be done to remove that namespace 
    # when the expression of a code statement is executed
    for name, ns_id in normal_nexts:
        name_watchers[(name.split('.')[-1], ns_id)] = (statement, normal_nexts, namespaces + [new_namespace | prev_namespace], promise)

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
            raise_error_at_line(filename, code, current_line, "Something went wrong with accessing the next value of a variable.") 
        mod_name = get_modified_next_name(name, id(ns))
        match old_len:
            case None: new_namespace[mod_name] = Name(mod_name, v.value if isinstance(v, Name) else v.prev_values[0])
            case i:
                if not isinstance(v, Variable):
                    raise_error_at_line(filename, code, current_line, "Something went wrong.")
                new_namespace[mod_name] = Name(mod_name, v.prev_values[i])
    return new_namespace

def interpret_name_watching_statement(statement: CodeStatementWithExpression, namespaces: list[Namespace], promise: Optional[DreamberdPromise], async_statements: AsyncStatements, when_statement_watchers: WhenStatementWatchers): 

    # evaluate the expression using the names off the top
    expr_val = evaluate_expression(statement.expression, namespaces, async_statements, when_statement_watchers) 
    index_vals = [evaluate_expression(expr, namespaces, async_statements, when_statement_watchers) 
                  for expr in statement.indexes] if isinstance(statement, VariableAssignment) else []
    namespaces.pop()  # remove expired namespace  -- THIS IS INCREDIBLY IMPORTANT

    match statement:
        case ReturnStatement():
            if promise is None:
                raise_error_at_line(filename, code, current_line, "Something went wrong.")
            promise.value = expr_val  # simply change the promise to that value as the return statement already returned a promise
        case VariableDeclaration():
            declare_new_variable(statement, expr_val, namespaces, async_statements, when_statement_watchers)
        case VariableAssignment():
            assign_variable(statement, index_vals, expr_val, namespaces, async_statements, when_statement_watchers)
        case Conditional():
            execute_conditional(expr_val, statement.code, namespaces, when_statement_watchers)
        case AfterStatement():
            execute_after_statement(expr_val, statement.code, namespaces, when_statement_watchers)
        case ExpressionStatement(): 
            print_expression_debug(statement.debug, statement.expression, expr_val, namespaces)

def clear_temp_namespace(namespaces: list[Namespace], temp_namespace: Namespace) -> None:
    for key in temp_namespace:
        del namespaces[-1][key]

# simply execute the conditional inside a new scope
def execute_conditional(condition: Value, statements_inside_scope: list[tuple[CodeStatement, ...]], namespaces: list[Namespace], when_statement_watchers: WhenStatementWatchers) -> Optional[Value]:
    condition = db_to_boolean(condition)
    execute = condition.value == True if condition.value is not None else random.random() < 0.50
    if execute:  
        return interpret_code_statements(statements_inside_scope, namespaces + [{}], [], when_statement_watchers + [{}]) # empty scope and async statements, just for this :)

# this is the equaivalent of an event listener
def get_mouse_event_object(x: int, y: int, button: mouse.Button, event: str) -> DreamberdObject:
    return DreamberdObject("MouseEvent", {
        'x': Name('x', DreamberdNumber(x)),
        'y': Name('y', DreamberdNumber(y)),
        'button': Name('button', DreamberdString(str(button).split('.')[-1])),
        'event': Name('event', DreamberdString(event)),
    })

def get_keyboard_event_object(key: Optional[Union[keyboard.Key, keyboard.KeyCode]], event: str) -> DreamberdObject:
    return DreamberdObject("MouseEvent", {
        'key': Name('key', DreamberdString(str(key).split('.')[-1])),
        'event': Name('event', DreamberdString(event)),
    })

def execute_after_statement(event: Value, statements_inside_scope: list[tuple[CodeStatement, ...]], namespaces: list[Namespace], when_statement_watchers: WhenStatementWatchers) -> None:

    if not KEY_MOUSE_IMPORTED:
        raise_error_at_line(filename, code, current_line, "Attempted to use mouse and keyboard functionality without importing the [input] extra dependency.")

    if not isinstance(event, DreamberdString):
        raise_error_at_line(filename, code, current_line, f"Invalid event for the \"after\" statement: \"{db_to_string(event)}\"")

    match event.value:
        case "mouseclick":
            mouse_buttons = {}
            def listener_func(x: int, y: int, button: mouse.Button, pressed: bool):
                nonlocal namespaces, statements_inside_scope
                if pressed:
                    mouse_buttons[button] = (x, y)
                else: 
                    if mouse_buttons[button]:   # it has been released and then pressed again
                        interpret_code_statements(statements_inside_scope, namespaces + [{'event': Name('event', get_mouse_event_object(x, y, button, event.value))}], [], when_statement_watchers + [{}])
                    del mouse_buttons[button]
            listener = mouse.Listener(on_click=listener_func)  # type: ignore
 
        case "mousedown":
            def listener_func(x: int, y: int, button: mouse.Button, pressed: bool):
                nonlocal namespaces, statements_inside_scope
                if pressed:
                    interpret_code_statements(statements_inside_scope, namespaces + [{'event': Name('event', get_mouse_event_object(x, y, button, event.value))}], [], when_statement_watchers + [{}])
            listener = mouse.Listener(on_click=listener_func)  # type: ignore

        case "mouseup":
            def listener_func(x: int, y: int, button: mouse.Button, pressed: bool):
                nonlocal namespaces, statements_inside_scope
                if not pressed:
                    interpret_code_statements(statements_inside_scope, namespaces + [{'event': Name('event', get_mouse_event_object(x, y, button, event.value))}], [], when_statement_watchers + [{}])
            listener = mouse.Listener(on_click=listener_func)  # type: ignore

        case "keyclick":
            keys = set()
            def on_press(key: Optional[Union[keyboard.Key, keyboard.KeyCode]]):
                nonlocal namespaces, statements_inside_scope
                keys.add(key)
            def on_release(key: Optional[Union[keyboard.Key, keyboard.KeyCode]]):
                nonlocal namespaces, statements_inside_scope
                if key in keys:
                    interpret_code_statements(statements_inside_scope, namespaces + [{'event': Name('event', get_keyboard_event_object(key, event.value))}], [], when_statement_watchers + [{}])
                keys.discard(key)
            listener = keyboard.Listener(on_press=on_press, on_release=on_release)  # type: ignore

        case "keydown":
            def on_press(key: Optional[Union[keyboard.Key, keyboard.KeyCode]]):
                nonlocal namespaces, statements_inside_scope
                interpret_code_statements(statements_inside_scope, namespaces + [{'event': Name('event', get_keyboard_event_object(key, event.value))}], [], when_statement_watchers + [{}])
            listener = keyboard.Listener(on_press=on_press)  # type: ignore

        case "keyup":
            def listener_func(x: int, y: int, button: mouse.Button, pressed: bool):
                nonlocal namespaces, statements_inside_scope
                if not pressed:
                    interpret_code_statements(statements_inside_scope, namespaces + [{'event': Name('event', get_mouse_event_object(x, y, button, event.value))}], [], when_statement_watchers + [{}])
            listener = keyboard.Listener(on_click=listener_func)  # type: ignore

        case _:
            raise_error_at_line(filename, code, current_line, f"Invalid event for the \"after\" statement: \"{db_to_string(event)}\"")

    listener.start()

def gather_names_or_values(expr: ExpressionTreeNode) -> set[Token]:
    names: set[Token] = set()
    match expr:
        case FunctionNode():
            for arg in expr.args:
                names |= gather_names_or_values(arg)
        case ListNode():
            for val in expr.values:
                names |= gather_names_or_values(val)
        case ExpressionNode():
            names |= gather_names_or_values(expr.right) | gather_names_or_values(expr.left)
        case IndexNode():
            names |= gather_names_or_values(expr.index) | gather_names_or_values(expr.value)
        case SingleOperatorNode():
            names |= gather_names_or_values(expr.expression)
        case ValueNode():
            names.add(expr.name_or_value)
    return names

def register_when_statement(condition: Union[list[Token], ExpressionTreeNode], statements_inside_scope: list[tuple[CodeStatement, ...]], namespaces: list[Namespace], async_statements: AsyncStatements, when_statement_watchers: WhenStatementWatchers):

    # if it is a variable, store it as the address to that variable.
    # if the internal value is a list, store it as an address to that mutable type.
    built_condition = get_built_expression(condition)
    gathered_names = gather_names_or_values(built_condition)
    caller_names = [n for name in gathered_names if (n := '.'.join(name.value.split('.')[:-1]))]
    dict_keys = [id(v) if isinstance(v := get_name_from_namespaces(name.value, namespaces), Variable) else name.value for name in gathered_names]\
                + [id(v.value) for name in gathered_names if (v := get_name_from_namespaces(name.value, namespaces)) is not None and isinstance(v.value, DreamberdMutable)]\
                + [id(v) for name in caller_names if isinstance(v := get_name_from_namespaces(name, namespaces), Variable)]\
                + [id(v.value) for name in caller_names if (v := get_name_from_namespaces(name, namespaces)) is not None and isinstance(v.value, DreamberdMutable)]\

    # the last comprehension watches callers of things (like list in list.length), and requires some implementation in the evaluate_expression function 
    # so that the caller of a function is also observed for it being called
    
    # register for future whens
    for name in dict_keys:
        if name not in when_statement_watchers[-1]:
            when_statement_watchers[-1][name] = []
        when_statement_watchers[-1][name].append((built_condition, statements_inside_scope))

    # check the condition now
    condition_value = evaluate_expression(built_condition, namespaces, async_statements, when_statement_watchers)
    execute_conditional(condition_value, statements_inside_scope, namespaces, when_statement_watchers)
    
def interpret_statement(statement: CodeStatement, namespaces: list[Namespace], async_statements: AsyncStatements, when_statement_watchers: WhenStatementWatchers) -> Optional[Value]:

    # build a list of expressions that are modified to allow for the next keyword
    expressions_to_check: list[Union[list[Token], ExpressionTreeNode]] = []
    match statement:
        case VariableAssignment(): expressions_to_check = [statement.expression] + statement.indexes
        case VariableDeclaration() | Conditional() | AfterStatement() | ExpressionStatement(): expressions_to_check = [statement.expression]
    all_normal_nexts: set[tuple[str, int]] = set()
    all_async_nexts: set[str] = set()
    next_filtered_exprs: list[ExpressionTreeNode] = []
    for expr in expressions_to_check:
        built_expr = get_built_expression(expr)
        new_expr, normal_nexts, async_nexts = handle_next_expressions(built_expr, namespaces)
        all_normal_nexts |= normal_nexts 
        all_async_nexts |= async_nexts
        next_filtered_exprs.append(new_expr)
    prev_namespace: Namespace = {}
    all_nexts = {s[0] for s in all_normal_nexts} | all_async_nexts
    for expr in next_filtered_exprs:
        prev_namespace |= save_previous_values_next_expr(expr, all_nexts, namespaces)

    # we replace the expression stored in each statement with the newly built one  -- TODO: consider changing this as to not override expressions
    match statement:
        case VariableAssignment():
            statement.expression = next_filtered_exprs[0]
            statement.indexes = next_filtered_exprs[1:]
        case VariableDeclaration() | Conditional() | AfterStatement() | ExpressionStatement():
            statement.expression = next_filtered_exprs[0]
    if all_normal_nexts:
        match statement:  # this is here to make sure everything is going somewhat smoothly
            case VariableAssignment() | VariableDeclaration() | Conditional() | AfterStatement() | ExpressionStatement(): pass
            case _:
                raise_error_at_line(filename, code, current_line, "Something went wrong. It's not your fault, it's mine.")
        adjust_for_normal_nexts(statement, all_async_nexts, all_normal_nexts, None, namespaces, prev_namespace)
        return   # WE ARE EXITING HERE!!!!!!!!!!!!!!!!!!!!!!!!!!

    if all_async_nexts:  # temporarily put prev_next_valeus (blah blah blah) into the namespace along with those that are generated from await next
        namespaces[-1] |= (prev_namespace := prev_namespace | wait_for_async_nexts(all_async_nexts, namespaces))

    # finally actually execute the damn thing
    retval = None
    match statement:
        case VariableAssignment():
            assign_variable(
                statement, [evaluate_expression(expr, namespaces, async_statements, when_statement_watchers) for expr in statement.indexes], 
                evaluate_expression(statement.expression, namespaces, async_statements, when_statement_watchers),
                namespaces, async_statements, when_statement_watchers
            )

        case FunctionDefinition():
            namespaces[-1][statement.name.value] = Name(statement.name.value, DreamberdFunction(
                args = [arg.value for arg in statement.args],
                code = statement.code,
                is_async = statement.is_async
            ))

        case DeleteStatement(): 
            val, ns = get_name_and_namespace_from_namespaces(statement.name.value, namespaces)
            if val and ns:
                del ns[statement.name.value]
            deleted_values.add(determine_non_name_value(statement.name))

        case ExpressionStatement():
            val = evaluate_expression(statement.expression, namespaces, async_statements, when_statement_watchers)
            print_expression_debug(statement.debug, statement.expression, val, namespaces)

        case Conditional():
            condition = db_to_boolean(evaluate_expression(statement.expression, namespaces, async_statements, when_statement_watchers))
            retval = execute_conditional(condition, statement.code, namespaces, when_statement_watchers)

        case AfterStatement():  # create event listener
            event = evaluate_expression(statement.expression, namespaces, async_statements, when_statement_watchers)
            execute_after_statement(event, statement.code, namespaces, when_statement_watchers)

        case WhenStatement():  # create variable listener  # one more left !!! :D
            register_when_statement(statement.expression, statement.code, namespaces, async_statements, when_statement_watchers)
            
        case VariableDeclaration():
            declare_new_variable(statement, evaluate_expression(
                statement.expression, namespaces, async_statements, when_statement_watchers
            ), namespaces, async_statements, when_statement_watchers)

        case ImportStatement():
            for name in statement.names:
                if not (v := importable_names.get(name.value)):
                    raise_error_at_token(filename, code, f"Name {name.value} could not be imported.", name)
                namespaces[-1][name.value] = Name(name.value, v)

        case ExportStatement():
            for name in statement.names:
                if not (v := get_name_from_namespaces(name.value, namespaces)):
                    raise_error_at_token(filename, code, "Tried to export name that is not a name or variable.", name)
                exported_names.append((statement.target_file.value, name.value, v.value))

        case ClassDeclaration(): 

            class_namespace: Namespace = {}
            fill_class_namespace(statement.code, namespaces, class_namespace, async_statements)

            # create a builtin function that is a closure returning an object of that class
            instance_made = False
            class_name = statement.name

            def class_object_closure(*args: Value) -> DreamberdObject:
                nonlocal instance_made, class_namespace, class_name
                if instance_made:
                    raise_error_at_line(filename, code, current_line, f"Already made instance of the class \"{class_name}\".")
                instance_made = True 
                obj = DreamberdObject(class_name.value, class_namespace)
                if constructor := class_namespace.get(class_name.value):
                    args = [obj] + list(args)  # type: ignore
                    if not isinstance(func := constructor.value, DreamberdFunction):
                        raise_error_at_line(filename, code, current_line, "Cannot create class variable with the same name as the class.")
                    if len(func.args) > len(args):
                        raise_error_at_line(filename, code, current_line, f"Expected more arguments for function call with {len(func.args)} argument{'s' if len(func.args) != 1 else ''}.")
                    new_namespace: Namespace = {name: Name(name, arg) for name, arg in zip(func.args, args)}
                    interpret_code_statements(func.code, namespaces + [new_namespace], [], when_statement_watchers + [{}])
                    del class_namespace[class_name.value]  # remove the constructor
                return obj

            namespaces[-1][statement.name.value] = Name(statement.name.value, BuiltinFunction(-1, class_object_closure))

    clear_temp_namespace(namespaces, prev_namespace)
    return retval

def fill_class_namespace(statements: list[tuple[CodeStatement, ...]], namespaces: list[Namespace], class_namespace: Namespace, async_statements: AsyncStatements) -> None:
    for possible_statements in statements:
        statement = determine_statement_type(possible_statements, namespaces)
        match statement:
            case FunctionDefinition():
                if "this" in statement.args:
                    raise_error_at_line(filename, code, current_line, "\"this\" keyword not allowed in class function declaration arguments.")
                class_namespace[statement.name.value] = Name(statement.name.value, DreamberdFunction(
                    args = ["this"] + [arg.value for arg in statement.args],    # i need to somehow make the this keyword actually do something
                    code = statement.code,
                    is_async = statement.is_async
                ))
            case VariableDeclaration(): 

                # why the frick are my function calls so long i really need some globals  
                var_expr = evaluate_expression(statement.expression, namespaces, async_statements, [{}]) 
                declare_new_variable(statement, var_expr, namespaces + [class_namespace], async_statements, [{}])  # don't want anything happening here, it's a different name
            case _:
                raise_error_at_line(filename, code, current_line, f"Unexpected statement of type {type(statement).__name__} in class declaration.")

# cleanup out of date variables
def decrement_variable_lifetimes(namespaces: list[Namespace]) -> None:
    for ns in namespaces: 
        remove_vars = []
        for name, v in ns.items():
            if not isinstance(v, Variable):
                continue 
            for l in v.lifetimes:
                l.lines_left -= 1 
            v.clear_outdated_lifetimes()
            if not v.lifetimes:
                remove_vars.append(name)
        for name in remove_vars:
            del ns[name]  # simply remove the variable from the name as it will be unable to provide a value

# change the current line number to some token in the code
def edit_current_line_number(statement: CodeStatement) -> None:
    global current_line
    match statement:
        case CodeStatementKeywordable(): 
            current_line = statement.keyword.line
        case FunctionDefinition():
            current_line = statement.keywords[0].line 
        case VariableDeclaration():
            current_line = statement.modifiers[0].line 
        case VariableAssignment():
            current_line = statement.name.line
        case ReturnStatement(): 
            if statement.keyword:
                current_line = statement.keyword.line
        case ExpressionStatement():
            if t := get_expr_first_token(get_built_expression(statement.expression)):
                current_line = t.line

# if a return statement is found, this will return the expression evaluated at the return. otherwise, it will return None
# this is done to allow this function to be called when evaluating dreamberd functions
def interpret_code_statements(statements: list[tuple[CodeStatement, ...]], namespaces: list[Namespace], async_statements: AsyncStatements, when_statement_watchers: WhenStatementWatchers) -> Optional[Value]:

    curr, direction = 0, 1
    while 0 <= curr < len(statements): 
        decrement_variable_lifetimes(namespaces)
        statement = determine_statement_type(statements[curr], namespaces)

        # if no statement found -i.e. the user did something wrong
        if statement is None:
            raise_error_at_line(filename, code, current_line, "Error parsing statement. Try again.")
        edit_current_line_number(statement)
        if isinstance(statement, ReturnStatement):  # if working with a return statement, either return a promise or a value
            expr, normal_nexts, async_nexts = handle_next_expressions(get_built_expression(statement.expression), namespaces)
            prev_namespaces = save_previous_values_next_expr(expr, async_nexts | {s[0] for s in normal_nexts}, namespaces)
            if normal_nexts:
                promise = DreamberdPromise(None)
                adjust_for_normal_nexts(statement, async_nexts, normal_nexts, promise, namespaces, prev_namespaces)
                return promise
            elif async_nexts:
                namespaces[-1] |= (prev_namespaces := prev_namespaces | wait_for_async_nexts(async_nexts, namespaces))
            retval = evaluate_expression(expr, namespaces, async_statements, when_statement_watchers)
            clear_temp_namespace(namespaces, prev_namespaces)
            return retval
        elif isinstance(statement, ReverseStatement):
            direction = -direction
        
        # otherwise interpret the other statement, if there is a return value then return
        elif retval := interpret_statement(statement, namespaces, async_statements, when_statement_watchers):
            return retval
        curr += direction

        # emulate taking turns running all the "async" functions
        for i in range(len(async_statements)):
            async_st, async_ns, line_num, async_direction = async_statements[i]
            if not 0 <= line_num < len(async_st):
                continue

            # this uesd to say async_st.pop(0) and i was genuinely wondering why it was changing, i'm so dumb
            statement = determine_statement_type(async_st[line_num], async_ns)
            if statement is None:
                raise_error_at_line(filename, code, current_line, "Error parsing statement. Try again.")
            elif isinstance(statement, ReturnStatement):
                raise_error_at_line(filename, code, current_line, "Function executed asynchronously cannot have a return statement.")
            elif isinstance(statement, ReverseStatement):
                async_direction = -async_direction
            async_statements[i] = (async_st, async_ns, line_num + async_direction, async_direction)  # messy but works
            interpret_statement(statement, async_ns, async_statements, when_statement_watchers)
    
# btw, reason async_statements and when_statements cannot be global is because they change depending on scope,
# due to (possibly bad) design decisions, the name_watchers does not do this... :D
def load_globals(_filename: str, _code: str, _name_watchers: NameWatchers, _deleted_values: set[Value], _exported_names: list[tuple[str, str, Value]], _importable_names: dict[str, Value]):
    global filename, code, name_watchers, deleted_values, current_line, exported_names, importable_names  # screw bad practice, not like anyone's using this anyways
    filename = _filename 
    code = _code
    name_watchers = _name_watchers 
    deleted_values = _deleted_values
    exported_names = _exported_names
    importable_names = _importable_names
    current_line = 1
    
def interpret_code_statements_main_wrapper(statements: list[tuple[CodeStatement, ...]], namespaces: list[Namespace], async_statements: AsyncStatements, when_statement_watchers: WhenStatementWatchers):
    try:
        interpret_code_statements(statements, namespaces, async_statements, when_statement_watchers)
    except NonFormattedError as e:
        raise_error_at_line(filename, code, current_line, str(e))
