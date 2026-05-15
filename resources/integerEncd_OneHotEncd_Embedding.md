Suppose your vocabulary is:

```python
words = ["cat", "dog", "fish"]
```

## 1) Integer encoding

Each word is assigned a unique integer ID.

```python
cat  -> 1
dog  -> 2
fish -> 3
```

So the sentence:

```python
"cat dog fish"
```

becomes:

```python
[1, 2, 3]
```

### Important

These numbers do **not** carry meaning.
`2` is not “twice as much” as `1`.
They are just labels.

---

## 2) One-hot encoding

Each word is converted into a vector with length equal to vocabulary size.

For the same vocabulary:

```python
cat  -> [1, 0, 0]
dog  -> [0, 1, 0]
fish -> [0, 0, 1]
```

So:

```python
"cat dog fish"
```

becomes:

```python
[[1, 0, 0],
 [0, 1, 0],
 [0, 0, 1]]
```

### What this means

* Only one position is `1`
* All others are `0`
* It does **not** capture meaning or similarity
* It only says “this word is different from the others”

---

## 3) Embedding

Embedding is a learned dense vector representation of a word.

Instead of a sparse vector like:

```python
cat -> [1, 0, 0]
```

the model learns something like:

```python
cat  -> [0.21, -0.44, 0.87]
dog  -> [0.18, -0.41, 0.80]
fish -> [-0.55, 0.12, 0.33]
```

### What this means

Words that appear in similar contexts can get similar vectors.

That is why embeddings are better than one-hot for language tasks.

---

# Comparison in one view

| Method           | Example for `cat`     |     Captures meaning? |                  Used in LSTM NLP? |
| ---------------- | --------------------- | --------------------: | ---------------------------------: |
| Integer encoding | `1`                   |                    No | Yes, usually as input to embedding |
| One-hot encoding | `[1, 0, 0]`           |                    No |         Sometimes, but inefficient |
| Embedding        | `[0.21, -0.44, 0.87]` | Yes, learned by model |            Yes, best common choice |

---

# Best practical pipeline for next-word prediction

For your project, the normal flow is:

```python
text -> tokenizer -> integer sequences -> padding -> embedding -> LSTM -> prediction
```

Example:

```python
sentence = "cat dog fish"
```

### Step 1: Tokenizer

```python
cat -> 1
dog -> 2
fish -> 3
```

### Step 2: Integer sequence

```python
[1, 2, 3]
```

### Step 3: Padding

If max length is 5:

```python
[0, 0, 1, 2, 3]
```

### Step 4: Embedding layer

The model turns each integer into a learned vector.

---

# Final correction to remember

* **Integer encoding** = word IDs
* **One-hot encoding** = sparse binary vectors
* **Embedding** = dense learned vectors

The most useful one for next-word prediction with LSTM is usually **integer encoding + embedding**.
