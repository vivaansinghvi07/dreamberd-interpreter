# DreamBerd Interpreter

This is the interpreter for the perfect programming language. It is made in Python, for the sole reason that the interpreter can itself be interpreted. Future plans include creating a DreamBerd interpreter in DreamBerd, so that the DreamBerd Interpreter can be passed into the DreamBerd Interpreter Interpreter, which is then interpreted by the DreamBerd Interpreter Interpreter Interpreter (a.k.a. Python). This may or may not be created due to difficulty moving everything over and whatnot. I'll try though.

This is incredibly slow. My implementation of DreamBerd is suboptimal, which itself runs on a subperformant language (Python), which runs on a pretty fast language (C). However, speed was never a focus in creating my interpreter for DreamBerd and shouldn't be - it's not a language meant for day-to-day use - it's a work of art.

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
  file                  the file containing your DreamBerd code

options:
  -h, --help            show this help message and exit
  -s, --show-traceback  show the full Python trackback upon errors
```

## TODO 

- Add another expression type which is just the dot operator, used for indexing and accessing names
- Better debugging (pretty limited for the time being)
- A much better standard library

## Absent Features

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

## Implemented Features

These are features that are implemented according to the [DreamBerd specification](https://github.com/TodePond/DreamBerd---e-acc) in this interpreter. 

### Exclamation Marks!

Be bold! End every statement with an exclamation mark!

```javascript
print("Hello world")!
```

If you're feeling extra-bold, you can use even more!!!

```javascript
print("Hello world")!!!
```

If you're unsure, that's ok. You can put a question mark at the end of a line instead. It prints debug info about that line to the console for you.

```javascript
print("Hello world")?
```

You might be wondering what DreamBerd uses for the 'not' operator, which is an exclamation mark in most other languages. That's simple - the 'not' operator is a semi-colon instead.

```javascript
if (;false) {
   print("Hello world")!
}
```

### Declarations

There are four types of declaration. Constant constants can't be changed in any way.

```javascript
const const name = "Luke"!
```

Constant variables can be edited, but not re-assigned.

```javascript
const var name = "Luke"!
name.pop()!
name.pop()!
```

Variable constants can be re-assigned, but not edited.

```javascript
var const name = "Luke"!
name = "Lu"!
```

Variable variables can be re-assigned and edited.

```javascript
var var name = "Luke"!
name = "Lu"!
name.push("k")!
name.push("e")!
```

### Immutable Data

**New for 2023!**<br>
Mutable data is an anti-pattern. Use the `const const const` keyword to make a constant constant constant. Its value will become constant and immutable, and will _never change_. Please be careful with this keyword, as it is very powerful, and will affect all users globally forever.

```javascript
const const const pi = 3.14!
```

#### Notes About Implementation

This is added by me (the interpreter)! I wanted to share how this works.

Thanks to [this repo](https://github.com/marcizhu/marcizhu) for helpful reference for issues and actions in Python.

To store public globals, the following steps are taken:
- On the user's side, open a GitHub issue with a title of the format `Create Public Global: {name};;;{confidence}` and the body containing the pickled version of the value.
- Then, run a GitHub workflow that puts the issue body into a file under `global_objects/` and add an entry to `public_globals.txt` that contains the `name;;;id;;;confidence`
- Finally, to retrieve these values, the content of each of these files is fetched and converted back into values.

### Naming

Both variables and constants can be named with any Unicode character or string.

```javascript
const const firstAlphabetLetter = 'A'!
var const 👍 = True!
var var 1️⃣ = 1!
```

This includes numbers, and other language constructs.

```javascript
const const unchanging = const!
unchanging unchanging 5 = 4!
print(2 + 2 === 5)! //true
```

### Arrays

Some languages start arrays at `0`, which can be unintuitive for beginners. Some languages start arrays at `1`, which isn't representative of how the code actually works. DreamBerd does the best of both worlds: Arrays start at `-1`.

```javascript
const const scores = [3, 2, 5]!
print(scores[-1])! //3
print(scores[0])!  //2
print(scores[1])!  //5
```

**New for 2022!**<br>
You can now use floats for indexes too!

```javascript
const var scores = [3, 2, 5]!
scores[0.5] = 4!
print(scores)! //[3, 2, 4, 5]
```

### When

In case you really need to vary a variable, the `when` keyword lets you check a variable each time it mutates.

```javascript
const var health = 10!
when (health = 0) {
   print("You lose")!
}
```

#### Technical Info 

Hi! It's me again. I took some creative liberty implementing the `when` statement, here's how it works:

- When defined, gather a list of names that are used in the expression of the statement.
- If a variable is detected, cause the when satement to watch that variable.
    - This is done in order to avoid watching names instead of variables when, say, a different variable with the same name is defined in a different scope.
    - Speaking of scope, when statements for which changes are detected in a different scope (from that of definition) **use that scope within their code**.
        - Looking back on my design decision, I am probably going to change this to make them always use the scope where they were defined.
- Additionally, if a variable detected contains a mutable value, that mutable value is also watched, so the following code detects a change:
    ```javascript
    const var l = [1, 2, 3]!
    when (l.length === 4) {
       print l!  
    }
    const var l_alias = l!
    l_alias[1.5] = 4!  // triggers the when statement
    ```

Therefore, the when statement can contain as complex an expression as desired. One small pitfall is that I've implemented it with recursion, which may cause performance issues (although I don't really care about performance, obvious in the fact that this is in Python).

### Lifetimes

DreamBerd has a built-in garbage collector that will automatically clean up unused variables (note: this is simply Python's garbage collector, I didn't implement anything). However, if you want to be extra careful, you can specify a lifetime for a variable, with a variety of units.

```javascript
const const name<2> = "Luke"! // lasts for two lines
const const name<20s> = "Luke"! // lasts for 20 seconds
```

By default, a variable will last until the end of the program. But you can make it last in between program-runs by specifying a longer lifetime.

```javascript
const const name<Infinity> = "Luke"! // lasts forever
```

> Yes, this is a thing. It stores your variables and values to a folder in your home directory.

### Loops

Loops are a complicated relic of archaic programming languages. In DreamBerd, there are no loops.

### Booleans

Booleans can be `true`, `false` or `maybe`.

```javascript
const var keys = {}!
after "keydown" { keys[event.key] = true! }
after "keyup" { keys[event.key] = false! }

function isKeyDown(key) => {
   if (keys[key] = undefined) {
      return maybe!
   }
   return keys[key]!
}
```

**Technical info:** Booleans are stored as one-and-a-half bits.

### Arithmetic

DreamBerd has significant whitespace. Use spacing to specify the order of arithmetic operations.

```javascript
print(1 + 2*3)! //7
print(1+2 * 3)! //9
```

Unlike some other languages, DreamBerd allows you to use the caret (^) for exponentiation.

```javascript
print(1^1)! // 1
print(2^3)! // 8
```

You can also use the number name, for example:

```javascript
print(one+two)! // 3
print  (twenty two  +  thirty three)!  // 55
```

> Yes, the second line is also valid. In an effort to preserve my sanity, I have limited this quirk to all numbers up to 99. After that, you're on your own.

### Indents

When it comes to indentation, DreamBerd strikes a happy medium that can be enjoyed by everyone: All indents must be 3 spaces long.

```javascript
function main() => {
   print("DreamBerd is the future")!
}
```

-3 spaces is also allowed.

```javascript
   function main() => {
print("DreamBerd is the future")!
   }
```

> Note: Your code will err if you have indents that are not a multiple of three.

### Equality

JavaScript lets you do different levels of comparison. `==` for loose comparison, and `===` for a more precise check. DreamBerd takes this to another level.

You can use `==` to do a loose check.

```javascript
3.14 == "3.14"! // true
```

You can use `===` to do a more precise check.

```javascript
3.14 === "3.14"! // false
```

You can use `====` to be EVEN MORE precise!

```javascript
const const pi = 3.14!
print(pi ==== pi)!  // true
print(3.14 ==== 3.14)!  // false
print(3.14 ==== pi)!  // false
```

If you want to be much less precise, you can use `=`.

```javascript
3 = 3.14! //true
```

### Functions

To declare a function, you can use any letters from the word `function` (as long as they're in order):

```javascript
function add (a, b) => a + b!
func multiply (a, b) => a * b!
fun subtract (a, b) => a - b!
fn divide (a, b) => a / b!
functi power (a, b) => a ** b!
union inverse (a) => 1/a!
```

### Dividing by Zero

Dividing by zero returns `undefined`.

```javascript
print(3 / 0)! // undefined
```

### Strings

Strings can be declared with single quotes or double quotes.

```javascript
const const name = 'Lu'!
const const name = "Luke"!
```

They can also be declared with triple quotes.

```javascript
const const name = '''Lu'''!
const const name = "'Lu'"!
```

In fact, you can use any number of quotes you want.

```javascript
const const name = """"Luke""""!
```

Even zero.

```javascript
const const name = Luke!
```

#### Technical Info

- To parse strings with many quotes, the interpreter scans the code for the shortest possible string.
- As soon as a pair of quote groups is found that is equal in terms of quote count on both sides, that is considered a string.
    - For example, `""""""` reads the two first double quotes, detects that there is a pair (`"` and `"`), and returns the corresponding empty string. This is repeated twice for the two remaining pairs of double quotes.
    - Therefore, to avoid premature detections of strings, simply create your starting quote with a single `'` and any number of `"`, like so: `'"""Hello world!'''''''`
- This is as complicated as it is in order to allow the declaration of empty strings without many problems.

### String Interpolation

Please remember to use your regional currency when interpolating strings.

```javascript
const const name = "world"!
print("Hello ${name}!")!
print("Hello £{name}!")!
print("Hello ¥{name}!")!
```

> Note: It was specified in the original repo to allow developers to follow their local typographical norms. While I think I could, that is not something I want to do and therefore I will not do it.

### Types

Type annotations are optional.

```javascript
const var age: Int = 28!
```
 
By the way, strings are just arrays of characters.

```javascript
String == Char[]!
```

Similarly, integers are just arrays of digits. Hello again! Because of this, you can index into integers! 

```javascript
const var my_number = 20!
my_number[-0.5] = 1!
print(my_number)!
```

If you want to use a binary representation for integers, `Int9` and `Int99` types are also available.

```javascript
const var age: Int9 = 28!
```

**Technical info:** Type annotations don't do anything, but they help some people to feel more comfortable.

### Previous

The `previous` keyword lets you see into the past!<br>
Use it to get the previous value of a variable.

```javascript
const var score = 5!
score = score + 1!
print(score)! // 6
print(previous score)! // 5
```

Similarly, the `next` keyword lets you see into the future!

```javascript
const var score = 5!
after ("click") { score = score + 1! }
print(await next score)! // 6 (when you click)
```

Additionally, the `current` keyword lets you see into the present!!

```javascript
const var score = 5!
print(current score)! // 5
```

### Exporting

Many languages allow you to import things from specific files. In DreamBerd, importing is simpler. Instead, you export _to_ specific files!

```java
===== add.db3 ==
function add(a, b) => {
   return a + b!
}

export add to "main.db3"!

===== main.db3 ==
import add!
add(3, 2)!
```

### Classes

You can make classes, but you can only ever make one instance of them. This shouldn't affect how most object-oriented programmers work.

```javascript
class Player {
   const var health = 10!
}

const var player1 = new Player()!
const var player2 = new Player()! // Error: Can't have more than one 'Player' instance!
```

This is how you could do this:

```javascript
class PlayerMaker {
   function makePlayer() => {
      class Player {
         const var health = 10!
      }
      const const player = new Player()!
      return player!
   }
}

const const playerMaker = new PlayerMaker()!
const var player1 = playerMaker.makePlayer()!
const var player2 = playerMaker.makePlayer()!
```

### Delete

To avoid confusion, the `delete` statement only works with primitive values like numbers, strings, and booleans (I actually decided to implement it to delete those and also non-primitive things like variables - really, anything in the namespace).

```javascript
delete 3!
print(2 + 1)! // Error: 3 has been deleted
```

DreamBerd is a multi-paradigm programming language, which means that you can `delete` the keywords and paradigms you don't like.

```javascript
delete class!
class Player {} // Error: class was deleted
```

When perfection is achieved and there is nothing left to `delete`, you can do this:

```javascript
delete delete!
```

### Overloading

You can overload variables. The most recently defined variable gets used.

```javascript
const const name = "Luke"!
const const name = "Lu"!
print(name)! // "Lu"
```

Variables with more exclamation marks get prioritised.

```javascript
const const name = "Lu"!!
const const name = "Luke"!
print(name)! // "Lu"

const const name = "Lu or Luke (either is fine)"!!!!!!!!!
print(name)! // "Lu or Luke (either is fine)"
```

### Reversing

You can reverse the direction of your code.

```javascript
const const message = "Hello"!
print(message)!
const const message = "world"!
reverse!
```

### Class Names

For maximum compatibility with other languages, you can alternatively use the `className` keyword when making classes.

This makes things less complicated.

```javascript
className Player {
   const var health = 10!
}
```

In response to some recent criticism about this design decision, we would like to remind you that this is part of the JavaScript specification, and therefore - out of our control.

### Semantic naming

DreamBerd supports semantic naming.

```javascript
const const sName = "Lu"!
const const iAge = 29!
const const bHappy = true!
```

**New for 2023:** You can now make globals.

```javascript
const const g_fScore = 4.5!  // Interpreter maker here... idk if this is supposed to do anything, I could implement this easily if I had to
```

### Asynchronous Functions

In most languages, it's hard to get asynchronous functions to synchronise with each other. In DreamBerd, it's easy: Asynchronous functions take turns running lines of code.

```javascript
async funct count() {
   print(1)!
   print(3)!
}

count()!
print(2)!
```

You can use the `noop` keyword to wait for longer before taking your turn.

```javascript
async func count() {
   print(1)!
   noop!
   print(4)!
}

count()!
print(2)!
print(3)!
```

**Note:** In the program above, the computer interprets `noop` as a string and its sole purpose is to take up an extra line. You can use any string you want.

### Signals

To use a signal, use `use`.

```javascript
const var score = use(0)!
```

When it comes to signals, the most important thing to discuss is _syntax_.

In DreamBerd, you can set (and get) signals with just one function:

```javascript
const var score = use(0)!

score(9)! // Set the value
score()?  // Get the value (and print it)
```

### Copilot

It's worth noting that Github Copilot doesn't understand DreamBerd, which means that Microsoft won't be able to steal your code.

This is great for when you want to keep your open-sourced project closed-source.

### Highlighting

Syntax highlighting is now available for DreamBerd in VSCode. To enable it, install a [highlighting extension](https://marketplace.visualstudio.com/items?itemName=fabiospampinato.vscode-highlight) and then use the [DreamBerd configuration file](https://github.com/TodePond/DreamBerd/blob/main/.vscode/settings.json).

This is what it looks like:

```
const const name = "Luke"!
print(name)! // "Luke"
```

**Please note:** The above code will only highlight correctly if you have the extension installed.

### Parentheses

Wait, I almost forgot!

Parentheses in DreamBerd do nothing. They get replaced with whitespace.<br>
The following lines of code all do the same thing.

```javascript
add(3, 2)!
add 3, 2!
(add (3, 2))!
add)3, 2(!
```

Lisp lovers will love this feature. Use as many parentheses as you want!

```javascript
(add (3, (add (5, 6))))!
```

Lisp haters will also love it.

```javascript
(add (3, (add (5, 6)!
```

Due to certain design decisions, `"("` is replaced with `" "` while `")"` is replaced with `""`.
