import json
import dataclasses
from typing import Any, Union, assert_never
from dreamberd.base import NonFormattedError

from dreamberd.builtin import BuiltinFunction, DreamberdFunction, DreamberdList, DreamberdNumber, DreamberdString, Name, Value, Variable

SerializedDict = dict[str, Union[str, dict, list]]

def serialize_obj(obj: Any) -> SerializedDict:
    match obj:
        case Name() | Variable() | Value(): return serialize_dreamberd_obj(obj)
        case _: return serialize_python_obj(obj)

def serialize_python_obj(obj: Any) ->  dict[str, Union[str, dict, list]]:
    match obj:
        case dict(): 
            if not all(isinstance(k, str) for k in obj):
                raise NonFormattedError("Serialization Error: Encountered non-string dictionary keys.")
            val = {k: serialize_obj(v) for k, v in obj.items()}
        case list(): val = [serialize_obj(x) for x in obj]
        case str(): val = obj
        case int() | float(): val = str(obj)
        case _: assert_never(obj)
    return {
        "python_obj_type": type(obj).__name__,
        "value": val
    }

def serialize_dreamberd_obj(val: Union[Value, Name, Variable]) -> dict[str, Union[str, dict, list]]:
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
    list_test_case = DreamberdList([
        DreamberdString("Hello world!"),
        DreamberdNumber(123.45)
    ])
    __import__('pprint').pprint(json.dumps(serialize_obj(list_test_case)))
        
