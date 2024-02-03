__doc__ = """
This script takes an issue containing:
    1) A variable name in the title 
    2) Pickled data in the form of a string of three-digit numbers (representing bytes)
And does the following:
    1) Create an entry for the data in a file with the name pointing to the respective ID 
    2) Create a file in the `stored_objects` directory with the name of the ID and the information in the variable
"""
import os 
import random

if __name__ == "__main__":

    NAME_TO_ID_SEP = ";;;"
    VARIABLE_NAME, CONFIDENCE = os.environ["ISSUE_TITLE"].removeprefix("Create Public Global: ").split(NAME_TO_ID_SEP)
    ISSUE_BODY = os.environ["ISSUE_BODY"]

    value_id = -1
    while value_id < 0 or value_id in os.listdir("global_objects"):
        value_id = random.randint(1, 1000000000)

    with open(f"./stored_objects/{value_id}.py") as f:
        f.write(ISSUE_BODY)
    with open(f"./public_globals.txt") as f:
        f.write(f"{VARIABLE_NAME}{NAME_TO_ID_SEP}{value_id}{NAME_TO_ID_SEP}{CONFIDENCE}\n")
