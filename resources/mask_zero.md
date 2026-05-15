Because your sequences are padded with zeros.

You did:

```python id="1h81w6"
pad_sequences(..., padding='pre')
```

Example:

Suppose max length is 6.

A short sequence:

```text id="wfuh0m"
[12, 45, 90]
```

becomes:

```text id="91b6x0"
[0, 0, 0, 12, 45, 90]
```

Those leading zeros are NOT real words.

They are artificial padding tokens added only to make all sequences the same length.

---

Without:

```python id="mvr1ad"
mask_zero=True
```

the embedding layer treats:

```text id="8vwy84"
0
```

like a real vocabulary word.

So the model starts learning embeddings for padding itself.

That is bad because:

* padding has no semantic meaning
* LSTM wastes capacity modeling fake tokens
* gradients become noisier
* learning quality drops

---

With:

```python id="y84rv6"
mask_zero=True
```

Keras creates a mask:

```text id="xchmfu"
0 -> ignore
nonzero -> process normally
```

So the LSTM skips padded positions entirely.

---

Example:

Without masking:

```text id="twv0kz"
[0, 0, 0, 12, 45, 90]
```

LSTM sees:

* timestep 1 → token 0
* timestep 2 → token 0
* timestep 3 → token 0
* timestep 4 → token 12
  ...

This pollutes sequence learning.

---

With masking:

LSTM effectively sees only:

```text id="jlwm18"
[12, 45, 90]
```

which is what you actually want.

---

Why this matters MORE for LSTMs than dense models:

LSTMs process sequences timestep-by-timestep.

So fake padding tokens at the beginning can distort:

* hidden states
* memory cell updates
* temporal dependencies

Masking prevents that.

---

One subtle but important thing:

Because you use:

```python id="jlwm27"
padding='pre'
```

the zeros are at the START.

Without masking, the LSTM spends early timesteps processing garbage.

That is especially harmful.

---

So in your setup:

```python id="jlwm35"
Embedding(..., mask_zero=True)
```

is the correct choice.
