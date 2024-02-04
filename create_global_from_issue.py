__doc__ = """
This script takes an issue containing:
    1) A variable name in the title 
    2) Pickled data in the form of a string of three-digit numbers (representing bytes)
And does the following:
    1) Create an entry for the data in a file with the name pointing to the respective ID 
    2) Create a file in the `stored_objects` directory with the name of the ID and the information in the variable
"""
import os 
import github

if __name__ == "__main__":

    with github.Github(auth=github.Auth.Token(os.environ["HOME_TOKEN"])) as g:
        issue = g.get_repo("vivaansinghvi07/dreamberd-interpreter").get_issue(int(os.environ["ISSUE_NUMBER"]))
        issue.edit(state='closed')

    with github.Github(auth=github.Auth.Token(os.environ["GLOBALS_REPO_TOKEN"])) as g:
        g.get_repo("vivaansinghvi07/dreamberd-interpreter-globals").create_issue(os.environ["ISSUE_TITLE"], os.environ["ISSUE_BODY"])
