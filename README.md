# DreamBerd Interpreter

This is the interpreter for the perfect programming language. It is made in Python, for the sole reason that the interpreter can itself be interpreted. Future plans include creating a DreamBerd interpreter in DreamBerd, so that the DreamBerd Interpreter can be passed into the DreamBerd Interpreter Interpreter, which is then interpreted by the DreamBerd Interpreter Interpreter Interpreter (a.k.a. Python).

## Installation

You can install DreamBerd from PyPi, by doing any the following:

```
$ pip install dreamberd 
$ pip install "dreamberd[input, globals]"
$ pip install "dreamberd[input]"
$ pip install "dreamberd[globals]"
```

Each of these commands installs DreamBerd with the respective dependencies. `input` installs the `pynput` package and allows the use of `after` statements and event watchers. `globals` installs `PyGithub` and allows you to declare `const const const` variables that are publically stored using GitHub. Note: to use the latter, you must enter a [Personal Access Token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens) in the `GITHUB_ACCESS_TOKEN` environment variable.

## Usage

Now that you have installed DreamBerd, you can run the REPL using the `$ dreamberd` command, or you can run a file using `$ dreamberd FILE`. Usage instructions here:

```
usage: dreamberd [-h] [-s] [file]

positional arguments:
  file                  The file containing your DreamBerd code.

options:
  -h, --help            show this help message and exit
  -s, --suppress-traceback
                        Limit the error traceback to a single message.
```

## TODO 

- Add different files exporting and importing things across ?
- Add string interpolation
  - Use the locale module like this:
  ```python3
  locale.setlocale(locale.LC_ALL, locale.getdefaultlocale()[0])
  locale.localeconv()['currency_symbol']
  ```
- Better debugging (pretty limited for the time being)
- Signals (not too hard)
- A much better standard library

## Features

The goal of this project is to implement every feature from the DreamBerd language. A list of features is in the README file of the project, linked [here](https://github.com/TodePond/DreamBerd---e-acc). Here is a working list of features that there is no chance I will implement (new features may be added - or I should say, removed - as I work on this project and realize I'm too stupid to implement them):

- DB3X: I am not going to even try to parse XML AND parse DB code.
- Regex: Since type hints seem to not even do anything there is no point in implementing a Regex parser. 
- "Variable Hoisting" (being able to declare variables with a negative lifetime): Given the fact that keywords can be renamed and reassigned in this language, it does not make sense to implement this as the following breaks:

    ```javascript
    print(name)
    var const = "lol";
    const const name<-2> = "Jake";
    ```
    It is impossible to evaluate the expression on the right side of the `name` declaration after the print statement. Additionally, doing so doesn't account for possible renaming of keywords in the second line.
- Any sort of autocomplete requires more brainpower than I am willing to put in.

To my knowledge, everything else has been or will be implemented.

### Storing Public Global Variables 

Thanks to [this repo](https://github.com/marcizhu/marcizhu) for helpful reference for issues and actions in Python.

To store public globals, the following steps are taken:
- On the user's side, open a GitHub issue with a title of the format `Create Public Global: {name};;;{confidence}` and the body containing the pickled version of the value.
- Then, run a GitHub workflow that puts the issue body into a file under `global_objects/` and add an entry to `public_globals.txt` that contains the `name;;;id;;;confidence`
- Finally, to retrieve these values, the content of each of these files is fetched and converted back into values.
