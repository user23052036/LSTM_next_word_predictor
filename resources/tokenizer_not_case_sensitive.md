Yes — and that is intentional behavior.

By default, Keras `Tokenizer` converts everything to lowercase before building the vocabulary.

So these are treated identically:

```python
"The Teacher Is On The Box"
"the teacher is on the box"
"THE TEACHER IS ON THE BOX"
```

All become internally:

```python
"the teacher is on the box"
```

That is why token indices stay the same.

Example:

```python
from tensorflow.keras.preprocessing.text import Tokenizer

texts = ["Hello world"]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

print(tokenizer.word_index)
```

Output:

```python
{'hello': 1, 'world': 2}
```

Now:

```python
tokenizer.texts_to_sequences(["HELLO WORLD"])
```

still gives:

```python
[[1, 2]]
```

because tokenizer lowercases input automatically.

---

Why this design exists:

In NLP, usually:

```text
Apple
apple
APPLE
```

represent the same semantic word most of the time.

Lowercasing:

* reduces vocabulary size
* improves generalization
* reduces sparsity
* helps small datasets

Otherwise vocabulary explodes unnecessarily.

---

But this has tradeoffs.

You lose information when capitalization matters:

```text
apple  -> fruit
Apple  -> company
US     -> country abbreviation
us     -> pronoun
```

So in tasks like:

* Named Entity Recognition
* Legal documents
* Biomedical NLP
* Case-sensitive code models

lowercasing can hurt performance.

---

You can disable this behavior:

```python
Tokenizer(lower=False)
```

Then:

```python
"Hello"
"hello"
```

become different tokens.

---

One more subtle thing most beginners miss:

Tokenizer also removes punctuation by default.

So:

```python
"hello!"
"hello"
```

often become the same token too.

That is controlled by:

```python
filters=
```

parameter.
