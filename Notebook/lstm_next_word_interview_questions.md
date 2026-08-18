# LSTM Next-Word Prediction Project — Interview Questions and Answers

> **How to use this document:** Every claim is grounded in what the notebooks actually execute. Commented-out code is labeled explicitly. No metrics, dataset sizes, or hyperparameters are invented.

---

## 1. Project Overview

This project implements an end-to-end next-word prediction system using an LSTM neural network trained on a small, manually curated dataset of 173 English sentences. The pipeline covers dataset construction, tokenization, n-gram prefix sequence generation, pre-padding, one-hot encoding, model architecture design, hyperparameter grid search, training, artifact saving, and greedy-decoding inference.

**Key verified facts from the notebooks:**
- Dataset: 173 manually written sentences (hardcoded in the LSTM notebook; *not* loaded from file in the final run)
- Vocabulary: 653 unique words
- Max sentence length: 26 words (computed via `split(' ')`)
- Total prefix sequences: 1,717
- X shape: (1717, 25) — all sequences padded to max_length − 1
- Y shape: (1717, 654) — one-hot encoded over vocab_size + 1 classes
- Model: Embedding(654, 50) → LSTM(50, dropout=0.5) → Dense(654, softmax)
- Total parameters: 86,254
- Training: 100 epochs, batch size 64, validation_split=0.2
- Best validation accuracy achieved: **0.1453** (14.53%)
- Inference: greedy decoding (argmax), 10-word generation loop
- Saved artifacts: `next_word_model.keras`, `tokenizer.pkl`, `max_length.txt`

---

## 2. 30-Second Explanation

**How I would answer in an interview:**

> "I built a next-word prediction model using an LSTM. I started with a small dataset of 173 English sentences, tokenized them with Keras's Tokenizer, generated all prefix subsequences, padded them, and trained a three-layer model — an embedding layer, an LSTM, and a softmax output — to predict the next word. At inference, I feed a seed phrase and greedily generate the next word by taking the argmax of the model's output probability distribution."

---

## 3. 1-Minute Explanation

**How I would answer in an interview:**

> "The project has two main parts. First, I built a dataset of 173 sentences covering everyday topics. Second, I trained an LSTM to predict the next word in a sequence. The pipeline works like this: each sentence is tokenized into integer word IDs; I then generate all growing prefixes of each tokenized sentence and pad them to a fixed length of 25 tokens. The last token of each padded sequence is the target label, one-hot encoded over 653 vocabulary positions plus one padding class. The model learns to map a 25-token prefix to a probability distribution over the vocabulary. At inference, I take my seed text, tokenize and pad it, run it through the model, and take the highest-probability word as the next word, then feed the expanded string back in for the next prediction."

---

## 4. 2-Minute Technical Explanation

**How I would answer in an interview:**

> "The LSTM notebook hardcodes 173 English sentences as the dataset. I used Keras's `Tokenizer()` — without any `num_words` cap or OOV token — fitted on all sentences, which produced 653 unique word types. Index 0 is reserved for padding. For each sentence I generate all possible prefix sequences starting from length 2, giving 1,717 sequences total. These are padded with zeros on the left (`padding='pre'`) to length 26 — the longest sentence — and then split: the first 25 positions form X and the 26th position (the last token) forms Y, one-hot encoded to 654 dimensions. The model is a Sequential stack: an Embedding layer with `input_dim=654`, `output_dim=50`, `mask_zero=True`; a single LSTM layer with 50 units and `dropout=0.5`; and a Dense layer with 654 outputs and softmax activation. I compiled with categorical cross-entropy loss and Adam. Before building this final model I ran a grid search over dropout rates [0.2, 0.3, 0.4, 0.5], embedding dimensions [10, 15, 23, 25, 30], and LSTM dimensions [10, 15, 20, 25, 30] for 90 epochs each. The final model used 50-dimensional embeddings and 50 LSTM units — values *not* present in the grid search ranges — trained for 100 epochs. The best validation accuracy across all runs was 14.53%, a number the interviewer should know signals substantial overfitting and an extremely difficult prediction task even on training data."

---

## 5. Complete Project Pipeline

```
173 hand-written English sentences (hardcoded in LSTM notebook)
        ↓
Keras Tokenizer().fit_on_texts(dataset)
        ↓
653 unique word types; word_index maps word → integer (1–653)
        ↓
For each sentence: generate all prefix subsequences of lengths 2, 3, …, N
        ↓
1,717 variable-length integer sequences
        ↓
pad_sequences(sequence, maxlen=26, padding='pre')   → shape (1717, 26)
        ↓
X = padded[:, 0:-1]   → shape (1717, 25)
Y = padded[:, -1]     → shape (1717,)  integer labels
        ↓
Y = to_categorical(Y, num_classes=654)  → shape (1717, 654)
        ↓
Embedding(input_dim=654, output_dim=50, mask_zero=True)
        ↓
LSTM(50, dropout=0.5)
        ↓
Dense(654, activation='softmax')
        ↓
model.fit(X, Y, epochs=100, batch_size=64, validation_split=0.2)
        ↓
model.save("next_word_model.keras")
pickle.dump(tokenizer)  /  write max_length.txt
        ↓
Inference: tokenize seed → pad to maxlen=25 → model.predict → np.argmax → word lookup
```

---

## 6. Dataset Generation Questions

### Q1. The dataset_generator notebook generates sentences, but what does the LSTM notebook actually use?

**Answer:**

The LSTM notebook does **not** read from `../Data/dataset.txt`. The code to read the file is commented out in cell 1 of the LSTM notebook:

```python
# with open('../Data/dataset.txt','r') as sentence:
#     dataset = sentence.read().splitlines()
```

Instead, the dataset is a **hardcoded Python list of 173 manually written sentences** defined directly in the LSTM notebook. The `dataset_generator.ipynb` is a separate, independent experiment that was not actually used in the final LSTM training pipeline.

**How I would answer in an interview:**

> "The dataset generator notebook was an experiment I ran to explore synthetic data generation. However, the actual LSTM was trained on a hardcoded list of 173 real English sentences I wrote manually. The file-reading code in the LSTM notebook is commented out."

**Likely follow-up:** Isn't that a disconnect between your two notebooks?

**Follow-up answer:** Yes, and it is a real weakness. The two notebooks are separate experiments. The dataset_generator.ipynb was not wired into the LSTM pipeline. If I were redoing this, I would either use the generator's output or remove it to avoid confusion.

---

### Q2. What is `itertools.product` and why was it used in the dataset generator?

**Answer:**

`itertools.product` computes the Cartesian product of multiple iterables. Given two lists of length M and N, it produces M×N ordered pairs — every possible combination of one element from each list.

In the generator, it creates all (noun, adjective), (noun, verb), (noun, place), etc. pairs, turning word lists into sentences programmatically. For example, 10 nouns × 10 adjectives = 100 sentences of the form "the {noun} is {adjective}".

**How I would answer in an interview:**

> "`product()` gives me the full Cartesian product. With 10 nouns and 10 adjectives I get all 100 (noun, adjective) pairs automatically, so I can define a pattern once and get all combinations without writing nested loops."

**Likely follow-up:** What is combinatorial explosion and why does it matter here?

**Follow-up answer:** If I add a third list of 10 elements, combinations jump from 100 to 1,000. With four 10-element lists it is 10,000. The generator uses up to three-way products in pattern 10 (`n1 × n2 × adjectives`), which produces `10 × 9 × 10 = 900` sentences just for that one pattern. This is how 10 lists of 10 words produce 1,630 sentences total.

---

### Q3. How many sentences does the dataset_generator actually write to `dataset.txt`, and is there a bug?

**Answer:**

**Yes, there is a confirmed bug.** The execution order in the notebook is:

```python
random.seed(13)
random.shuffle(dataset)   # cell exec order 86: shuffles all 1,630 sentences

with open('../Data/dataset.txt','w') as file:   # cell exec order 93: writes ALL 1,630 sentences
    for item in dataset:
        file.write(item + "\n")

dataset = dataset[:500]   # cell exec order 66: truncates in memory ONLY
```

The `dataset[:500]` truncation **runs before the file write** in cell execution number (`execution_count=66` vs `execution_count=93`), but the truncation only affects the in-memory list. The file write at `execution_count=93` happens **after** the shuffle at `execution_count=86` but iterates over the full shuffled dataset, which at that point in the notebook is still 1,630 sentences.

Therefore `dataset.txt` contains **all 1,630 shuffled sentences**, not 500. The `assert len(dataset) == 500` at execution_count=66 validates the in-memory list *after* truncation, but this has no effect on what was written to disk.

**How I would answer in an interview:**

> "This is a real bug in my notebook. The file write happens before the assert and truncation in notebook execution order, so the file gets all 1,630 sentences. The 500-sentence limit only applies to the in-memory variable after that point. The comment in the code itself says 'this unnecessarily shuffles the full dataset where we just need to pick random 500 sentences' and suggests `random.sample(dataset, 500)` as a better approach."

**What the interviewer is testing:** Careful reading of execution order vs. cell order in Jupyter notebooks.

---

### Q4. What does `add_unique()` do and what is inefficient about it?

**Answer:**

```python
def add_unique(target_dataset, new_items):
    for item in new_items:
        item = " ".join(item.split())
        if item not in target_dataset:
            target_dataset.append(item)
```

It normalizes whitespace using `" ".join(item.split())` — `str.split()` without arguments splits on any whitespace and discards empty strings, then `" ".join()` reassembles with single spaces. It then checks for duplicates before appending.

**Inefficiency:** `if item not in target_dataset` is O(N) for each of the M new items, giving O(M×N) overall for each call. A `set` lookup is O(1), so a `set` for membership tracking would reduce this to O(M+N). The tradeoff is that a set does not preserve insertion order (though Python 3.7+ dicts do), and the code needs a separate list for ordered output.

**How I would answer in an interview:**

> "The whitespace normalization ensures consistent formatting regardless of how sentences are constructed. The duplicate check uses a list, which is O(N) per lookup. A better approach would be to maintain a parallel set for O(1) membership checks while keeping the list for order, or use a `dict.fromkeys()` pattern."

---

### Q5. What does `random.seed(13)` do and why does it matter?

**Answer:**

`random.seed(13)` initializes Python's pseudo-random number generator to a fixed state. Any subsequent call to `random.shuffle()` will produce the same permutation every time the cell is run. This ensures reproducibility — anyone running the notebook gets the same shuffled dataset.

**`random.shuffle()` vs `random.sample()`:**
- `shuffle()` operates in-place on the existing list. Time complexity O(N). Requires the full list to exist first.
- `sample(dataset, 500)` returns a new list of 500 randomly chosen items without replacement. It is O(500) instead of O(1630). More memory-efficient for large datasets when only a subset is needed.

The notebook comment acknowledges this: `# this unnecessarily shuffles the full dataset where we just need to pick random 500 sentences`.

---

### Q6. What sentence patterns does the dataset generator create?

**Answer (from actual code):**

| Pattern | Template | Combinations |
|---|---|---|
| 1. Description | `the {noun} is {adjective}` | 10×10 = 100 |
| 2. Capability | `the {noun} can {verb}` | 10×10 = 100 |
| 3. Location | `the {noun} is in the {place}` | 10×10 = 100 |
| 4a. Position-on | `the {noun} is on the {object}` | 10×10 = 100 |
| 4b. Position-under | `the {noun} is under the {object}` | 10×10 = 100 |
| 5. Relation | `the {n1} is near the {n2}` (n1≠n2) | 10×9 = 90 |
| 6. Preference | `{pronoun} like the {object}` | 4×10 = 40 |
| 7. Eating | `{pronoun} eat {food}` | 4×10 = 40 |
| 8. Drinking | `{pronoun} drink {food_subset}` | 4×5 = 20 |
| 9. State | `we/they are {adj}` + `i am {adj}` + `you are {adj}` | 20+10+10 = 40 |
| 10. Compound | `the {n1} and the {n2} are {adj}` (n1≠n2) | 90×10 = 900 |

Total before deduplication: ~1,630 unique sentences.

---

## 7. Synthetic Data Questions

### Q7. Why did you use a synthetic dataset instead of Shakespeare or TinyStories?

**Answer:**

The notebook contains the explanation as a comment:

```python
# this dataset is too complex to work with, so we will use a smaller dataset
```

Both Shakespeare (commented out) and TinyStories (commented out) were considered and rejected. The rationale was that a large real-world corpus introduces: a very large vocabulary (thousands of words), long and complex sentences, and an LSTM that would take much longer to train and show meaningful results. The synthetic dataset makes the learning problem tractable on a small machine without a GPU cluster.

**NOT IMPLEMENTED:** Reading from Shakespeare or TinyStories. These were only commented experiments.

**How I would answer in an interview:**

> "I considered both Shakespeare and TinyStories but commented them out. The rationale was controllability — with 173 sentences and 653 words I can observe the full training loop in seconds, understand what the model learned, and debug easily. For a learning project that is a reasonable tradeoff, though I fully acknowledge the resulting model cannot generalize to real language."

**Likely follow-up:** Does high accuracy on this dataset mean anything?

**Follow-up answer:** Very little. The dataset has low lexical diversity, and the many prefix sequences from a single sentence share almost all their context. A model that memorizes frequent transition patterns in this corpus will score well on validation sequences that are continuations of training sentences.

---

### Q8. What are the fundamental limitations of templated synthetic data for language modeling?

**Answer:**

1. **Missing linguistic diversity:** All sentences in the generator follow rigid Subject-Verb-Object or Noun-Adjective patterns. No questions, passives, subordinate clauses, or complex syntax.
2. **Template memorization vs. language learning:** The model does not learn general language; it learns to complete templates. "the cat is ____" always gets an adjective; the model learns the template, not semantics.
3. **Artificial transition probabilities:** In real language, "the" is followed by thousands of different word types. In templated data, "the cat is" is always followed by an adjective — the conditional distribution is artificially narrow.
4. **Zero OOV robustness:** The dataset vocabulary is the entire training vocabulary. There is no held-out test distribution to evaluate OOV handling.
5. **No noise:** Real language has punctuation, contractions, misspellings. The model is never prepared for any of that.

---

## 8. Tokenization Questions

### Q9. What is tokenization and why is it necessary?

**Answer:**

Tokenization converts raw text into integer indices. Neural networks cannot process strings directly — they require fixed-size numerical inputs. Each unique word is assigned a unique integer; the model learns to map these integers (via an embedding layer) to dense vector representations.

**How I would answer in an interview:**

> "An LSTM operates on tensors of numbers. 'the' means nothing to it — but index 1 does. Tokenization builds a vocabulary dictionary and converts every word to its integer ID so the model can process it mathematically."

**Likely follow-up:** What does `fit_on_texts()` actually do?

**Follow-up answer:** It scans all the sentences, counts word frequencies, sorts words by frequency (most frequent gets index 1), and builds `word_index` and `index_word` dictionaries. It also records `word_counts` (raw frequency) and `document_count` (number of sentences seen).

---

### Q10. Why does Keras Tokenizer index from 1, not 0? What is index 0 reserved for?

**Answer:**

Index 0 is reserved for the **padding token**. When sequences of different lengths are padded to the same length with `pad_sequences`, the padding value is 0. If words started at index 0, the model could not distinguish between a real word and a padding position. The `Embedding` layer with `mask_zero=True` uses this convention: index 0 is masked out and ignored in computations.

**How I would answer in an interview:**

> "Keras reserves index 0 for padding. My sequences are padded with zeros, and `mask_zero=True` on the embedding layer tells the LSTM to ignore those positions. If words started at 0, the padding would look like a real word to the model."

---

### Q11. What is `oov_token` and is it used in this project?

**Answer:**

The `oov_token` is a special token that the Tokenizer assigns to words not seen during training at inference time. In the notebook, the OOV configuration is commented out:

```python
# tokenizer = Tokenizer(num_words=500, oov_token='<nothing>')
tokenizer = Tokenizer()
```

**Not used.** The plain `Tokenizer()` silently drops unknown words — `texts_to_sequences` returns an empty list for completely unknown text, or a partial list if only some words are unknown. This is demonstrated in the notebook with `tokenizer.texts_to_sequences(['the stupid teacher keeps smoking cigeratte'])`, which returns `[[1, 49, 303]]` — only the three words in the vocabulary ('the', 'teacher', 'smoking') are kept; 'stupid', 'keeps', and 'cigeratte' are silently dropped.

**How I would answer in an interview:**

> "I do not use an OOV token. Unknown words at inference time are simply dropped by the tokenizer. This is a significant limitation — if someone types a word not in my 653-word vocabulary, it disappears silently and the prediction is based on reduced context."

---

### Q12. What is `index_word` and how is it used in inference?

**Answer:**

`tokenizer.index_word` is the reverse dictionary: integer index → word string. It is used in the inference loop to convert the model's predicted integer index back to a human-readable word:

```python
for word, index in tokenizer.word_index.items():
    if index == pos:
        text = text + " " + word
        break
```

**Inefficiency note:** This is O(V) per prediction step — it scans the entire word_index dictionary to find the word for a given index. The correct approach is to use `tokenizer.index_word[pos]` directly, which is an O(1) dict lookup. This is a code-quality issue that an interviewer may notice.

---

## 9. Vocabulary Questions

### Q13. What is vocab_size in this project and why is `+1` used everywhere?

**Answer:**

```python
vocab_size = len(tokenizer.word_index)  # = 653
```

`vocab_size = 653`. However, `vocab_size + 1 = 654` is used in:
- `Embedding(input_dim=vocab_size+1, ...)` — because the embedding table must have a row for index 0 (the padding position), so it needs 654 rows (indices 0 through 653).
- `Dense(vocab_size+1, ...)` — the output must be a probability over 654 classes (0 through 653).
- `to_categorical(Y, num_classes=vocab_size+1)` — Y vectors must be 654-dimensional.

**How I would answer in an interview:**

> "The +1 accounts for index 0, which is the padding token. My vocabulary has 653 real words (indices 1–653) and one padding slot (index 0). Every layer that touches indices needs to accommodate all 654 possible values."

---

### Q14. The comment in the vocab_size cell says "here we have 78 unique words." Is this accurate?

**Answer:**

**No. This is an inaccurate comment.** The actual output of `len(tokenizer.word_index)` is **653**, confirmed by the cell's output. The comment "# here we have 78 unique words" is leftover from a much earlier experiment with a smaller dataset and was never updated. This is a code maintenance issue.

**How I would answer in an interview:**

> "That comment is wrong — it is a stale comment from an earlier experiment. The actual vocabulary size is 653 unique words, as shown by the cell output. I should have updated or deleted that comment."

---

## 10. Sequence Generation Questions

### Q15. How are prefix sequences generated and what is the result?

**Answer:**

```python
sequence = []
for sentence in dataset:
    tokenized_sentence = tokenizer.texts_to_sequences([sentence])[0]
    for i in range(1, len(tokenized_sentence)):
        sequence.append(tokenized_sentence[:i+1])
```

For a sentence of length N words, this generates N-1 prefix sequences of lengths 2, 3, 4, …, N. For example, "the sky is blue and" generates:
- `[1, 27]` — "the sky"
- `[1, 27, 7]` — "the sky is"
- `[1, 27, 7, 148]` — "the sky is blue"
- `[1, 27, 7, 148, 2]` — "the sky is blue and"

The 173 sentences produce **1,717 total prefix sequences** (verified in cell output: `len(sequence) = 1717`).

**How I would answer in an interview:**

> "I expand each sentence into all possible prefixes. For a 5-word sentence I get 4 training examples. Each example teaches the model: given these N words, the next word is this. This is the standard approach for training next-word predictors with LSTMs."

**Likely follow-up:** Does this create leakage between training and validation?

**Follow-up answer:** Yes — this is a critical weakness. All prefixes of the same sentence are adjacent in the sequence list. When `validation_split=0.2` takes the last 20% of the list, it takes the end of the dataset, not randomly sampled sequences. Prefixes from the same sentence can fall on both sides of the split boundary. The model sees the early prefixes of a sentence in training and is then validated on later prefixes of the same sentence — which are not truly unseen.

---

## 11. Padding Questions

### Q16. Why is pre-padding used and what does it look like?

**Answer:**

```python
pad_sequences = pad_sequences(sequence, maxlen=max_length, padding='pre')
```

Pre-padding adds zeros at the **left** side of short sequences so all sequences reach length 26 (max_length). For example, `[1, 27]` becomes `[0, 0, 0, …, 0, 1, 27]` — 24 leading zeros.

Pre-padding is preferred for recurrent models because the meaningful content arrives at the end of the sequence, right before the LSTM produces its final hidden state used for prediction. Post-padding would force the LSTM to process meaningful words first and then many zeros, degrading its memory of the actual context by the time it outputs.

**How I would answer in an interview:**

> "Pre-padding places the actual words at the tail of the sequence, immediately before the LSTM makes its prediction. The LSTM reads the zeros first — which `mask_zero=True` causes it to skip — and then processes the real words. This preserves recency of context in the final hidden state."

---

### Q17. What is `mask_zero=True` doing and why does it matter?

**Answer:**

`mask_zero=True` in the `Embedding` layer propagates a mask through the network. Positions where the input is 0 (padding positions) are marked as masked. The LSTM layer respects this mask and does not update its hidden state for masked time steps. This prevents the padded zeros from influencing the model's predictions.

Without masking, the LSTM would process padding tokens as if they were real input, corrupting its hidden state with meaningless content.

**Not implemented:** The code does not explicitly check whether masking is correctly propagated through all layers, but TensorFlow's Sequential API propagates masks automatically from `Embedding` through `LSTM`.

---

### Q18. What is the shape of X after padding and why is it (1717, 25) rather than (1717, 26)?

**Answer:**

After `pad_sequences(..., maxlen=max_length)` the shape is (1717, 26). Then:

```python
X = pad_sequences[:, 0:-1]   # shape (1717, 25) — all columns except last
Y = pad_sequences[:, -1]     # shape (1717,)  — last column only
```

The 26th column is the target word (Y). X contains the 25-token context. The model's `Input(shape=(25,))` reflects this.

**Note on variable naming:** The code reuses the name `pad_sequences` for the result of calling `pad_sequences()`, which shadows the imported function. After this assignment, the original `pad_sequences` function is no longer accessible by that name. This is a Python naming hazard.

---

## 12. X and Y Questions

### Q19. Is there a shuffling step between creating X/Y and training? Should there be?

**Answer:**

**No shuffling is applied.** The code for shuffling X and Y together is commented out:

```python
# idx = np.random.permutation(len(X))
# X = X[idx]
# Y = Y[idx]
```

This matters because `validation_split=0.2` in Keras takes the **last 20%** of X and Y in their current order. Since the sequences are ordered by sentence (all prefixes of sentence 1, then all prefixes of sentence 2, etc.), the validation set consists of the last ~343 rows — which are the prefixes of the final ~34 sentences in the dataset. This is not a random sample of the full distribution; it is a biased tail sample.

**How I would answer in an interview:**

> "I did not shuffle X and Y before splitting. This means my validation set is not a random sample — it is the tail of the ordered sequence list, which represents only the last sentences in the dataset. A proper approach would be to shuffle X and Y together before calling `model.fit`, or use a manual train/val split with `sklearn.model_selection.train_test_split`."

---

### Q20. How is Y one-hot encoded and what does a Y vector look like?

**Answer:**

```python
Y = to_categorical(Y, num_classes=vocab_size+1)
```

Each integer label is converted to a binary vector of length 654. The position equal to the word's index is set to 1, all others are 0. For the first training example, the target is "sky" (index 27), so `Y[0]` is a length-654 vector with a 1 at position 27 and 0 everywhere else — confirmed in the cell output.

**Memory note:** One-hot encoding is expensive. With 1,717 examples and 654 classes, Y is a (1717, 654) float32 array ≈ 4.5 MB. For larger vocabularies this becomes impractical; sparse categorical cross-entropy avoids materializing the full one-hot matrix.

---

## 13. Embedding Questions

### Q21. What is an Embedding layer and what does it learn?

**Answer:**

An embedding layer is a trainable lookup table with shape `(vocab_size+1, embedding_dim)`. Each word index maps to one row — a dense vector of floating-point numbers. These vectors are initialized randomly and updated during backpropagation to capture distributional semantic relationships (words appearing in similar contexts get similar vectors).

In this project: `Embedding(input_dim=654, output_dim=50, mask_zero=True)` — 654 rows × 50 dimensions = 32,700 parameters (confirmed in model summary).

**How I would answer in an interview:**

> "The embedding layer converts discrete integer word IDs into continuous dense vectors. Each word gets a 50-dimensional representation. These vectors are learned end-to-end — the model adjusts them to minimize prediction loss, so words that tend to appear in similar contexts develop similar vectors."

**Likely follow-up:** Why not use pre-trained embeddings like GloVe or Word2Vec?

**Follow-up answer (honest):** Not implemented. Pre-trained embeddings would be beneficial — they provide semantic knowledge from large corpora. With only 173 training sentences, the model cannot learn meaningful semantic relationships from scratch. Using GloVe or Word2Vec embeddings (frozen or fine-tuned) would be a significant improvement.

> **Not used in my current implementation; interviewer may ask this as a conceptual follow-up.**

---

### Q22. Why 50 embedding dimensions?

**Answer:**

The notebook does not document the original rationale. The hyperparameter grid search swept [10, 15, 23, 25, 30] — **50 is not in the swept range.** The final model uses 50-dimensional embeddings, chosen independently of the grid search results.

**How I would answer in an interview:**

> "The notebook does not prove the original reason. The grid search swept embedding dimensions up to 30. The final model uses 50, which is larger than any value in the sweep. A defensible interview explanation would be: 50 is a standard rule-of-thumb for moderate vocabulary sizes and was chosen to give the model sufficient expressiveness without excessive parameters. However, I should honestly acknowledge that this value was not systematically validated."

---

## 14. LSTM Questions

### Q23. What is an LSTM and why was it chosen over a vanilla RNN?

**Answer:**

An LSTM (Long Short-Term Memory) is a recurrent neural network variant that adds a cell state and three gating mechanisms (input gate, forget gate, output gate) to control information flow. Vanilla RNNs suffer from vanishing gradients — gradients shrink to near zero as they are backpropagated through many time steps, making it impossible to learn long-range dependencies. LSTMs mitigate this by providing an additive gradient path through the cell state.

**How I would answer in an interview:**

> "I chose LSTM over vanilla RNN because my sequences can be up to 25 tokens long. Vanilla RNNs struggle to remember information from early in the sequence by the time they process the end. The LSTM's gating mechanisms allow it to selectively retain and update information across long sequences."

**Not implemented:** GRU, BiLSTM, Transformer. These are not used.

> **GRU, BiLSTM, Transformer — not used in my current implementation; interviewer may ask this as a conceptual follow-up.**

---

### Q24. Why is `return_sequences=False` (the default) used?

**Answer:**

`return_sequences=False` means the LSTM outputs only the **final hidden state** — a single vector of 50 values — rather than the hidden state at every time step. This is correct for next-word prediction: we want a single representation of the full context, not a sequence of representations.

The notebook has a commented-out stacked LSTM configuration:
```python
# LSTM(64, return_sequences=True, dropout=0.3),
```
This was considered but not implemented. For a stacked LSTM, the first layer would need `return_sequences=True` to feed its full output sequence to the second layer. This is not active in the final model.

---

### Q25. Why `dropout=0.5` in the LSTM?

**Answer:**

Dropout randomly sets a fraction of LSTM input connections to zero during training, acting as a regularizer to reduce overfitting. The value 0.5 was the highest value tested in the hyperparameter grid search. Looking at the sweep results, `Dropout: 0.5, Embedding: 30, LSTM: 30 --> Best Val Acc: 0.1453` was one of the better configurations. The final model uses dropout=0.5.

**How I would answer in an interview:**

> "The grid search included dropout values of 0.2, 0.3, 0.4, and 0.5. The 0.5 dropout configurations tended to produce comparable or slightly higher validation accuracy in many cases. Given the small dataset and high risk of overfitting, stronger regularization seemed justified. The final model uses 0.5."

---

## 15. LSTM Mathematics

### Q26. What are the four LSTM equations?

**Answer:**

Given input $$x_t$$, previous hidden state $$h_{t-1}$$, and previous cell state $$C_{t-1}$$:

$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
$$

$$
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
$$

$$
\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
$$

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

$$
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
$$

$$
h_t = o_t \odot \tanh(C_t)
$$

Where:
- $$f_t$$ = forget gate: how much of the old cell state to keep
- $$i_t$$ = input gate: how much of the new candidate to add
- $$\tilde{C}_t$$ = candidate cell state
- $$C_t$$ = updated cell state
- $$o_t$$ = output gate: how much of the cell to expose as hidden state
- $$h_t$$ = hidden state (output of the LSTM at time t)
- $$\sigma$$ = sigmoid activation
- $$\odot$$ = element-wise multiplication

---

### Q27. What is the vanishing gradient problem and how does the LSTM cell state address it?

**Answer:**

In vanilla RNNs, the gradient at time step t involves a product of many Jacobian matrices:

$$
\frac{\partial L}{\partial h_0} = \prod_{t=1}^{T} \frac{\partial h_t}{\partial h_{t-1}}
$$

If these Jacobians have eigenvalues < 1, the product shrinks exponentially. The LSTM's cell state update is additive:

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

The additive path through $$C_t$$ allows gradients to flow backward without being multiplied by a weight matrix at every step, greatly reducing (though not eliminating) the vanishing gradient problem.

---

## 16. Parameter Count Questions

### Q28. How many parameters does the model have and how are they distributed?

**Answer (verified from model.summary() output):**

| Layer | Output Shape | Parameters |
|---|---|---|
| Embedding | (None, 25, 50) | 32,700 |
| LSTM | (None, 50) | 20,200 |
| Dense | (None, 654) | 33,354 |
| **Total** | | **86,254** |

**Embedding:** `654 × 50 = 32,700`

**LSTM:** The LSTM has 4 gates, each with weights for input and hidden state plus biases:

$$
4 \times (50 \times 50 + 50 \times 50 + 50) = 4 \times (2500 + 2500 + 50) = 4 \times 5050 = 20,200
$$

Wait — input dimension is 50 (from embedding), hidden dimension is 50:

$$
4 \times (50 \times 50 + 50 \times 50 + 50) = 4 \times (2500 + 2500 + 50) = 20,200
$$

**Dense:** `50 × 654 + 654 = 32,700 + 654 = 33,354`

**Total: 86,254** — confirmed by model.summary().

**How I would answer in an interview:**

> "The model has about 86,000 parameters. The embedding table dominates with roughly 33,000, the LSTM has 20,200 from its four gate matrices, and the output Dense layer has another 33,000 for projecting from 50 hidden units to 654 output classes."

---

## 17. Softmax Questions

### Q29. Why is softmax used in the output layer?

**Answer:**

Next-word prediction is a multi-class classification problem with 654 classes. Softmax converts the raw logits from the Dense layer into a valid probability distribution — all values between 0 and 1, summing to 1:

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{654} e^{z_j}}
$$

This allows the model's output to be interpreted as "probability of each word being next," and the loss function (categorical cross-entropy) to measure how well this distribution matches the true one-hot target.

**How I would answer in an interview:**

> "I need a probability distribution over 654 possible next words. Softmax ensures all 654 outputs are positive and sum to 1, which is exactly what categorical cross-entropy expects as input."

---

## 18. Loss Function Questions

### Q30. Why categorical cross-entropy?

**Answer:**

Categorical cross-entropy measures the divergence between the true one-hot distribution and the model's predicted distribution:

$$
\mathcal{L} = -\sum_{c=1}^{654} y_c \log(\hat{y}_c)
$$

Since $$y_c = 1$$ for only one class (the true next word) and 0 for all others, this simplifies to:

$$
\mathcal{L} = -\log(\hat{y}_{\text{true}})
$$

It penalizes the model heavily when it assigns low probability to the correct next word. It is the standard loss for multi-class classification with one-hot targets and softmax output.

**Sparse categorical cross-entropy** would be equivalent but accepts integer labels (0–653) directly instead of one-hot vectors, avoiding the memory cost of materializing the (1717, 654) Y matrix. Not used in this project.

---

## 19. Adam Optimizer Questions

### Q31. Why Adam and what does it do?

**Answer:**

Adam (Adaptive Moment Estimation) maintains per-parameter estimates of the first moment (mean of gradients) and second moment (uncentered variance of gradients):

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$

$$
\theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
$$

Default values: $$\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-7}$$. The project uses these defaults.

Adam adapts the learning rate per parameter, making it robust to sparse gradients and less sensitive to hyperparameter tuning than SGD. It is the de facto default for training LSTMs.

---

## 20. Dropout Questions

### Q32. Where is dropout applied and what is its effect?

**Answer:**

In this project, `dropout=0.5` is passed directly to the `LSTM` layer constructor. In Keras, this applies dropout to the **input connections** of the LSTM (i.e., the input-to-gate connections at each time step), not the recurrent connections.

The recurrent dropout (dropout on the hidden-state-to-gate connections) would be specified with `recurrent_dropout=`. This is **not used** in the project.

**How I would answer in an interview:**

> "I pass `dropout=0.5` to the LSTM layer, which applies 50% dropout to the input connections at each time step during training. Recurrent dropout — on the hidden state connections — is not applied. Adding recurrent dropout would provide additional regularization."

---

## 21. Hyperparameter Tuning Questions

### Q33. How was hyperparameter tuning done and what is the critical mismatch?

**Answer:**

A grid search was run over:
- `dropout_rate = [0.2, 0.3, 0.4, 0.5]`
- `embedding_dim = [10, 15, 23, 25, 30]`
- `lstm_dim = [10, 15, 20, 25, 30]`

This is `4 × 5 × 5 = 100` combinations, each trained for **90 epochs** with `batch_size=64, validation_split=0.2`. The best validation accuracy for each configuration was printed.

**The critical mismatch:** The final model uses:
- Embedding dimension: **50** (not in the swept range of [10, 15, 23, 25, 30])
- LSTM dimension: **50** (not in the swept range of [10, 15, 20, 25, 30])
- Training epochs: **100** (not 90 as in the sweep)

The final model's hyperparameters were **not selected from the grid search results**. The grid search was effectively a learning exercise, and the final architecture was chosen independently with larger values.

**How I would answer in an interview:**

> "I ran a grid search over embedding dimensions up to 30 and LSTM sizes up to 30, for 90 epochs each. However, my final model uses 50 for both, which is outside the searched range. The grid search did not directly inform the final model choice — this is a weakness I would correct by including 50 in the search grid or running a follow-up comparison."

---

### Q34. What do the hyperparameter tuning results actually show?

**Answer:**

Looking at the actual results, validation accuracy ranges from about **0.1163 to 0.1453** across all 100 configurations. This is a very narrow and uniformly poor range. The best configuration (`Dropout: 0.5, Embedding: 30, LSTM: 30 --> Best Val Acc: 0.1453`) barely outperforms the worst (`0.1163`). This suggests the model is fundamentally limited by the dataset, not by hyperparameters. The hyperparameter sensitivity is very low — the dataset and validation methodology are the bottleneck, not the model architecture.

---

## 22. Training and Validation Questions

### Q35. Is there a true held-out test set?

**Answer:**

**No.** The project has:
- **Training data:** 80% of the 1,717 sequences (approximately 1,374 sequences)
- **Validation data:** 20% of the 1,717 sequences (approximately 343 sequences) — the last 20% in order, not randomly sampled
- **Test set:** None

There is no held-out test set that was withheld before training began. The validation set is also used to observe training progress, which in principle could lead to indirect overfitting if hyperparameters are selected based on it. Without a true test set, generalization cannot be measured.

---

### Q36. What does the training curve tell us?

**Answer (from actual training output):**

The training shows severe overfitting:
- Training accuracy improves steadily: from 6.9% (epoch 1) to 36.3% (epoch 100)
- Validation accuracy peaks early (~14.5% around epoch 27–39) and then **degrades** as training accuracy continues to rise
- By epoch 100, val_accuracy is only 10.5% while train_accuracy is 36.3%

The validation loss diverges massively from training loss (training loss ~2.85 at epoch 100, validation loss ~7.67). This is a textbook overfitting pattern. The model memorizes the training sequences but cannot generalize even to the (leaky) validation split.

---

### Q37. Why was early stopping not used?

**Answer:**

Early stopping was considered but **not implemented**. The code exists as commented-out cells:

```python
# from tensorflow.keras.callbacks import EarlyStopping
# early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
# callbacks=[early_stop]  # commented out in model.fit
```

Given the training curve, early stopping would have halted training around epoch 21–27 when validation accuracy was at its peak (~14.5%), and would have prevented the additional 73 epochs of overfitting.

**How I would answer in an interview:**

> "I considered early stopping — the code is commented out — but did not enable it in the final training run. In hindsight this was a mistake. The training curve shows validation accuracy peaks around epoch 27 and degrades for the remaining 73 epochs. Early stopping would have saved compute and returned a better-regularized model."

---

## 23. Data Leakage and Evaluation Risks

### Q38. What are the sequence-level leakage risks in this project?

**Answer:**

**Multiple leakage risks exist:**

1. **Prefix leakage:** All prefix sequences from one sentence share subsets of each other. Sentence 1 generates sequences `[t1, t2]`, `[t1, t2, t3]`, `[t1, t2, t3, t4]`, etc. All of these are present in the same ordered list. When `validation_split=0.2` takes the last 20%, early prefixes of the final sentences go to training and later prefixes go to validation. The model sees partial context in training and is validated on extended context from the same sentence.

2. **Vocabulary leakage:** The tokenizer is fitted on the entire dataset before splitting. The validation set has no OOV words relative to the model's vocabulary — in a proper setup, the tokenizer would be fitted only on training data.

3. **Split non-randomness:** The last 20% of sequences come from the last ~34 sentences. If those sentences share common words and patterns with the first 80%, this is not a real generalization test.

**How I would answer in an interview:**

> "There are real leakage issues. The most significant is that prefix sequences from the same sentence appear in both training and validation sets, because I did not shuffle before splitting and used a sequential 80/20 split. The tokenizer is also fitted on all data, creating vocabulary leakage. These issues mean the validation accuracy overestimates true generalization performance."

---

## 24. Metrics Questions

### Q39. Is accuracy a good metric for next-word prediction? What would be better?

**Answer:**

**Accuracy is a weak metric here.** It measures how often the model's top-1 prediction exactly matches the true next word. On a 654-class problem this is very stringent — a reasonable model that puts high probability on plausible alternatives will still score 0 for each wrong guess.

Better metrics:

- **Perplexity** — the exponentiated average negative log-likelihood per word. Lower is better. It measures how surprised the model is on average. It rewards distributing probability among plausible completions, not just the top guess.

$$
\text{Perplexity} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log P(w_i | w_1, \ldots, w_{i-1})\right)
$$

- **Top-5 accuracy** — whether the true word appears in the top-5 predictions.

> **Not used in my current implementation; perplexity and top-k accuracy are conceptual follow-ups.**

---

## 25. Inference and Text Generation

### Q40. Walk me through the inference procedure step by step.

**Answer (from actual code):**

```python
text = "the sun"

for i in range(10):
    token_text = tokenizer.texts_to_sequences([text])[0]
    padded_token_text = pad_sequences([token_text], maxlen=max_length-1, padding='pre')
    prediction = model.predict(padded_token_text, verbose=0)
    pos = np.argmax(prediction)
    for word, index in tokenizer.word_index.items():
        if index == pos:
            text = text + " " + word
            break
```

Step by step:
1. Start with seed text `"the sun"`
2. Tokenize: `texts_to_sequences(["the sun"])` → `[42]` (or `[1, 42]` — 'the'=1, 'sun'=42)
3. Pad left to length `max_length-1 = 25`
4. Feed to model: get probability vector of shape (1, 654)
5. `np.argmax(prediction)` → integer index of the highest-probability word
6. Linear scan through `word_index` to find the word for that index (inefficient — should use `index_word`)
7. Append word to text string
8. Repeat 10 times

**Actual output from the notebook:**
```
the sun is
the sun is the
the sun is the students
the sun is the students is
the sun is the students is the
the sun is the students is the dog
the sun is the students is the dog is
the sun is the students is the dog is important
the sun is the students is the dog is important for
the sun is the students is the dog is important for your
```

This output shows **degenerate repetition** — the model converges to predicting high-frequency words ('the', 'is') repeatedly, indicating it has not learned meaningful language structure.

---

### Q41. What is greedy decoding and what are its limitations?

**Answer:**

Greedy decoding always selects the single highest-probability word at each step (`np.argmax`). It is implemented in this project.

Limitations:
- **No diversity:** The same seed text always produces the same output.
- **Repetition:** High-frequency words ('the', 'is') dominate, creating repetitive loops as seen in the actual output.
- **Suboptimal sequences:** The globally best sequence is not necessarily found by always taking the local best word. Beam search explores multiple paths.

**Not implemented:**
> **Beam search, temperature sampling, top-k sampling, top-p sampling — not used in my current implementation; interviewer may ask these as conceptual follow-ups.**

---

### Q42. What is the inference preprocessing mismatch risk?

**Answer:**

During training, X has shape (1717, 25) — `max_length-1 = 25` columns. During inference:

```python
padded_token_text = pad_sequences([token_text], maxlen=max_length-1, padding='pre')
```

This correctly uses `max_length-1 = 25` as the padding target — matching the training input shape. If `max_length` were to differ between training and inference (e.g., if loaded from `max_length.txt` incorrectly), the model would receive inputs of the wrong size.

**Saving `max_length` to `max_length.txt` and reloading it addresses this.** However, the saved value is the raw `max_length = 26`, and inference uses `max_length-1 = 25`. This subtraction must be applied consistently at inference time or a mismatch occurs.

---

## 26. Model Saving and Loading Questions

### Q43. What artifacts are saved and how?

**Answer (from actual executed cells):**

```python
model.save("next_word_model.keras")       # TensorFlow SavedModel format

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)             # Python pickle — Keras Tokenizer object

with open("max_length.txt", "w") as f:
    f.write(str(max_length))              # writes "26" as a plain text integer
```

Three artifacts are saved: the model weights and architecture, the tokenizer vocabulary and configuration, and the max sequence length.

**Reloading:**
```python
max_length = int(f.read())
tokenizer = pickle.load(f)
model = load_model("next_word_model.keras")
```

**Risk:** `pickle` is not secure — loading a pickled tokenizer from an untrusted source can execute arbitrary code. For production, a JSON-serialized tokenizer is safer.

---

## 27. Reproducibility

### Q44. Is this project reproducible?

**Answer:**

**Partially.** Several things work against full reproducibility:

1. No global random seed is set for TensorFlow/NumPy in the LSTM notebook (`tf.random.set_seed()` is not called). GPU operations have non-determinism that seeds do not fully control.
2. The dataset generator uses `random.seed(13)` for Python's random module, but this only affects the Python random module, not NumPy or TensorFlow operations.
3. Keras's `validation_split` is deterministic (takes the last 20% sequentially), so the split itself is reproducible.
4. The hyperparameter grid search results may differ between runs due to weight initialization randomness.

**How I would answer in an interview:**

> "The data split is reproducible because validation_split always takes the last 20%. However, the training results are not fully reproducible because I did not set a TensorFlow random seed. Different runs may give slightly different validation accuracy values."

---

## 28. Code-Level Questions

### Q45. What code-quality issues exist in the notebooks?

**Answer:**

Confirmed issues (verified from code):

1. **Stale comment:** `# here we have 78 unique words` — actual vocab size is 653.
2. **Variable name collision:** `pad_sequences` is used both as the function name (imported) and as the variable name for the result. After `pad_sequences = pad_sequences(...)`, the function is shadowed.
3. **Inefficient word lookup at inference:** O(V) linear scan instead of O(1) `tokenizer.index_word[pos]`.
4. **Cell execution order vs. visual order:** In dataset_generator.ipynb, the dataset is written to file before the truncation `dataset[:500]` — but visually the truncation cell appears earlier in the notebook, creating confusion.
5. **Inaccurate docstring in comments:** The comment on the sequence generation loop says "always remember that the output is a list of lists" — correct but does not explain the actual prefix logic.
6. **Split(' ') vs split():** `len(item.split(' '))` uses explicit space delimiter, which may fail on multiple spaces or other whitespace. The `add_unique` function correctly uses `item.split()` (no argument) for normalization, but the length computation uses `split(' ')`.

---

## 29. Comparison Questions

### Q46. How does this LSTM approach compare to a Transformer-based approach for next-word prediction?

**Answer:**

| Aspect | This LSTM | Transformer (e.g., GPT) |
|---|---|---|
| Architecture | Sequential, processes tokens one at a time | Parallel, processes all tokens simultaneously |
| Long-range dependencies | Limited by vanishing gradients | Full attention over all positions |
| Training data needed | Can train on small data (with caveats) | Requires large corpora |
| Positional awareness | Implicit in recurrent state | Requires explicit positional encoding |
| Parallelism | Low (sequential by nature) | High |
| Inference | Fast per token | Can be slower (depends on context length) |

> **Transformer — not used in my current implementation; interviewer may ask this as a conceptual follow-up.**

---

## 30. Scenario-Based Questions

### Q47. If I type "the dog", what happens at inference?

**Answer:**

1. `tokenizer.texts_to_sequences(["the dog"])` → `[1, 54]` ('the'=1, 'dog'=54)
2. Padded to length 25: `[0, 0, ..., 0, 1, 54]` (23 leading zeros)
3. Model predicts a probability distribution over 654 classes
4. `np.argmax()` selects the highest-probability index
5. The word for that index is appended

Based on the training data patterns, words frequently following "the dog" in the dataset (like "barked", "ran", "wagged", "watched", "slept", "followed") would be candidates. However given the overfitting evidence, the model may default to high-frequency words like "is" or "was".

---

### Q48. What if I type a word not in the vocabulary?

**Answer:**

The tokenizer drops unknown words silently. `texts_to_sequences(["the quantum"])` → `[1]` — only 'the' is kept; 'quantum' is not in the vocabulary and is dropped without warning. The model then receives a very short sequence (just one word) padded to length 25 with 24 leading zeros, and makes a prediction based only on 'the'. This prediction is likely meaningless context-wise.

There is no OOV handling — no `<UNK>` token was configured. This is a significant real-world limitation.

---

## 31. Project Weaknesses

### Q49. What are the most serious weaknesses in this project?

**Answer (honest, verified from code):**

**Critical weaknesses:**

1. **No true test set.** Generalization cannot be measured. The reported 14.53% validation accuracy is the only metric, and the validation set is not a clean holdout.

2. **Sequence-level leakage.** Prefix sequences from the same sentence appear in both training and validation splits due to sequential (non-random) splitting.

3. **The dataset generator was not used.** The LSTM was trained on a manually hardcoded 173-sentence list, making the `dataset_generator.ipynb` notebook an orphaned experiment.

4. **Severe overfitting.** Training accuracy reaches 36% by epoch 100; validation accuracy peaks at 14.5% around epoch 27 and degrades to ~10.5% by epoch 100. The model memorizes training sequences but cannot generalize.

5. **Degenerate inference output.** The actual inference output loops on high-frequency words ("the sun is the students is the dog is important for your"), showing the model has not learned meaningful language.

6. **Final model hyperparameters not from the grid search.** The grid searched up to embedding=30 and LSTM=30; the final model uses 50 for both — never validated.

7. **Early stopping not used.** The model trained 73 epochs past its peak validation performance.

8. **No OOV handling.** Unknown words at inference time are silently dropped.

9. **Inefficient word lookup.** O(V) linear scan instead of O(1) dictionary lookup at inference.

10. **Stale, incorrect comment.** `# here we have 78 unique words` when actual vocab is 653.

---

## 32. Improvement Questions

### Q50. What are the most impactful improvements you would make?

**Answer:**

**Minimal improvements with material impact:**

1. **Shuffle X and Y before splitting.** One line: `idx = np.random.permutation(len(X)); X, Y = X[idx], Y[idx]`. Eliminates the sequential split bias immediately.

2. **Enable early stopping.** Uncomment the existing `EarlyStopping` callback. Would stop training around epoch 27 and save compute.

3. **Add a held-out test set.** Manually withhold 10–15 sentences before any processing begins.

4. **Fix the OOV handling.** Add `oov_token='<OOV>'` to the `Tokenizer()` constructor.

5. **Fix the inference word lookup.** Replace the linear scan with `tokenizer.index_word[pos]`.

6. **Use `sparse_categorical_crossentropy`.** Avoid materializing the full (1717, 654) one-hot Y matrix.

7. **Expand the dataset.** Even doubling to 300–400 sentences with more lexical diversity would help.

8. **Use pre-trained embeddings.** GloVe 50d would provide semantic initialization. The 653-word vocabulary is small enough to use directly.

---

## 33. Production Questions

### Q51. What would it take to put this model into production?

**Answer:**

**Not deployed. Conceptual answer only:**

> **Flask/FastAPI, Docker, cloud deployment — not implemented in my current project; interviewer may ask this as a conceptual follow-up.**

Minimum steps for production:
1. Wrap inference in a function that handles input preprocessing and output postprocessing consistently.
2. Serve via a REST API (Flask/FastAPI) that loads model, tokenizer, and max_length once at startup.
3. Add input validation (handle OOV, empty input, inputs longer than max_length).
4. Containerize with Docker.
5. Add monitoring for request latency and prediction diversity (catch degenerate loops).

**Honest limitation:** This model would not be useful in production. Its 14.5% validation accuracy, degenerate inference output, and inability to handle OOV words make it unsuitable for real users.

---

## 34. "Why did you choose…" Questions

### Q52. Why LSTM?

> "The notebook does not prove the original reason. A defensible interview explanation would be: LSTM is the standard architecture for sequence modeling tasks of this scale. It handles variable-length input via its recurrent state and mitigates the vanishing gradient problem that makes vanilla RNNs impractical for sequences of 5–25 tokens."

### Q53. Why word-level tokenization?

> "The notebook does not document the original reason. A defensible explanation: word-level tokenization is simpler, more interpretable, and appropriate for a small closed-vocabulary dataset. Character-level or subword tokenization would be overkill for 653 unique words and would require more training data to be effective."

### Q54. Why pre-padding?

> "Pre-padding ensures the meaningful tokens are at the right end of the sequence, immediately influencing the LSTM's final hidden state used for prediction. Post-padding would force the LSTM to process meaningful content first and then many padding tokens, degrading its memory of context."

### Q55. Why `mask_zero=True`?

> "To prevent padding tokens (index 0) from influencing the LSTM's hidden state. Without masking, the model wastes capacity learning to process meaningless zeros."

### Q56. Why one-hot encoding rather than sparse categorical crossentropy?

> "The notebook does not document the original reason. One-hot encoding was likely chosen for explicitness — it makes the target representation concrete. Sparse categorical cross-entropy would be the more memory-efficient choice: it accepts integer labels directly and avoids the (1717, 654) float32 matrix."

### Q57. Why softmax?

> "Next-word prediction is multi-class classification. Softmax is the standard output activation for this problem because it produces a valid probability distribution over all vocabulary classes."

### Q58. Why categorical cross-entropy?

> "It is the standard loss for multi-class classification with one-hot targets and softmax output. It penalizes the model for assigning low probability to the correct next word."

### Q59. Why Adam?

> "Adam is the default optimizer for LSTMs. It adapts learning rates per parameter and is generally more stable than SGD for recurrent networks without extensive learning rate tuning."

### Q60. Why dropout 0.5?

> "The hyperparameter sweep included [0.2, 0.3, 0.4, 0.5]. Higher dropout values tended to produce comparable validation accuracy on this small, repetitive dataset, suggesting strong regularization was appropriate. The final model uses 0.5."

### Q61. Why 50 embedding dimensions?

> "The notebook does not prove the original reason. The grid search only swept up to 30. The final value of 50 was not validated by the sweep. A defensible explanation: 50 is a common rule-of-thumb for moderate vocabulary sizes. For 653 words, 50 dimensions provides enough representational capacity without excessive parameters."

### Q62. Why 50 LSTM units?

> "Same caveat as embedding dimensions — 50 was not in the grid search range. A defensible explanation: 50 LSTM units gives 20,200 parameters, which seems proportionate to the task complexity. Larger units risk more overfitting on 173 sentences."

### Q63. Why 100 epochs?

> "The notebook does not prove the original reason. The grid search used 90 epochs. The final model used 100. Given that early stopping was not enabled, 100 epochs allowed the model to fully overfit, which is visible in the training curve. A better choice would have been to use early stopping."

### Q64. Why batch size 64?

> "The notebook does not document the original reason. With 1,374 training samples and batch size 64, there are approximately 22 gradient updates per epoch (matching the '22/22' shown in training output). Batch size 64 is a common default. Larger batches would mean fewer updates per epoch; smaller batches would be noisier."

### Q65. Why 20% validation split?

> "The notebook does not document the original reason. 80/20 is the standard default. With 1,717 sequences, 20% ≈ 343 validation samples, which is marginally sufficient to observe training trends, though the sequential split means these 343 samples are not representative of the full distribution."

---

## 35. Hard Interview Questions

### HQ1. The file `dataset.txt` contains how many sentences?

**What the interviewer is testing:** Whether you actually traced the execution order in your notebook, not just the visual cell order.

**Strong answer:** All 1,630 sentences. The shuffle (exec 86) runs before the file write (exec 93), and the truncation `dataset[:500]` (exec 66) runs before both in notebook execution but only affects the in-memory variable. The file was written from the full shuffled list.

**Weak answer:** 500 sentences, because I truncated to 500.

**Common trap:** Confusing notebook cell position with execution order. The truncation cell appears visually near the end but was executed (execution_count=66) before the shuffle (execution_count=86) and file write (execution_count=93).

**Follow-up question:** How would you fix this?

**Follow-up answer:** Write to file after truncation: `dataset = dataset[:500]; open(...).write(...)`. Or use `random.sample(dataset, 500)` which returns a new list of 500 without modifying the original.

---

### HQ2. What does your validation accuracy of 14.5% actually mean?

**What the interviewer is testing:** Whether you can interpret the metric honestly given the leakage and data issues.

**Strong answer:** It means the model correctly predicts the exact next word 14.5% of the time on the last 20% of prefix sequences, ordered sequentially. Due to sequence-level leakage (prefixes of the same sentence in both splits), vocabulary leakage (tokenizer fitted on all data), and non-random splitting, this number overestimates true generalization. On genuinely unseen sentences, performance would be much lower — possibly near random (1/653 ≈ 0.15%).

**Weak answer:** My model is 14.5% accurate at predicting the next word.

**Common trap:** Treating the validation accuracy as a reliable performance estimate.

**Follow-up question:** What is random baseline accuracy for this task?

**Follow-up answer:** 1/654 ≈ 0.15% for uniform random guessing. The model at 14.5% is far better than random on the (leaky) validation set, but the actual generalization gap is unknown.

---

### HQ3. Why does the inference output repeat "is the" so many times?

**What the interviewer is testing:** Understanding of greedy decoding, distribution collapse, and model failure modes.

**Strong answer:** This is a symptom of degenerate greedy decoding combined with an overfit model that predicts high-frequency words. 'the' (index 1) and 'is' (index 7) are the most common words in the dataset. After predicting 'is', the context "... is" most commonly leads to another noun phrase starting with 'the', creating a loop. Greedy decoding amplifies this because it always selects the single highest-probability word with no diversity mechanism.

**Weak answer:** The model needs more training.

**Common trap:** Thinking more training would help. The issue is structural — greedy decoding on a distribution that peaked on common words, plus overfitting.

**Follow-up question:** How would you fix this?

**Follow-up answer:** Temperature sampling (divide logits by T < 1 to sharpen, T > 1 to flatten), top-k sampling, or top-p (nucleus) sampling would introduce diversity. Beam search would find better complete sequences. These are **not implemented** in this project.

---

### HQ4. Count the LSTM parameters manually.

**What the interviewer is testing:** Can you compute LSTM parameter counts from first principles?

**Strong answer:**
An LSTM with input dimension d_in and hidden dimension d_h has 4 gates, each with:
- Input weight matrix: d_in × d_h
- Recurrent weight matrix: d_h × d_h
- Bias vector: d_h

Here d_in = 50 (embedding output), d_h = 50 (LSTM units):

$$
4 \times (d_{in} \times d_h + d_h \times d_h + d_h) = 4 \times (50 \times 50 + 50 \times 50 + 50) = 4 \times 5050 = 20{,}200
$$

Confirmed by model.summary(): **20,200**.

**Weak answer:** I don't know, the summary said 20,200.

**Common trap:** Forgetting the recurrent weight matrix (h → h) or the biases.

**Follow-up question:** If you doubled the LSTM units to 100, how many parameters would the LSTM layer have?

**Follow-up answer:** d_in stays 50 (embedding output unchanged), d_h = 100:
$$4 \times (50 \times 100 + 100 \times 100 + 100) = 4 \times (5000 + 10000 + 100) = 4 \times 15100 = 60{,}400$$

---

### HQ5. Your hyperparameter sweep never included embedding=50 or LSTM=50. How do you justify your final model?

**What the interviewer is testing:** Whether you can acknowledge a gap in your experimental design honestly.

**Strong answer:** I cannot fully justify it from the sweep results alone. The grid searched embedding dimensions [10, 15, 23, 25, 30] and LSTM units [10, 15, 20, 25, 30]. The trend in the results suggests larger values tend to produce marginally better validation accuracy, but the differences are small (~1–3%). I extrapolated this trend to 50, but did not validate it. A rigorous approach would have included 50 in the sweep or run a direct comparison between the best grid-search configuration and the 50/50 final model.

**Weak answer:** 50 seemed like a good number.

**Common trap:** Claiming the sweep informed the final model when it objectively did not.

**Follow-up question:** What would you have done differently?

**Follow-up answer:** Include values up to 100 in the sweep, or at minimum add 50 as a candidate. Also match the final training epochs (100) to the sweep training epochs (90) to ensure comparability.

---

### HQ6. The inference code uses `maxlen=max_length-1`. Why not `max_length`?

**What the interviewer is testing:** Understanding of the X/Y split and the training input shape.

**Strong answer:** Training X has shape (1717, 25) = (1717, max_length-1). The model was compiled with `Input(shape=(25,))`. At inference, I must feed a sequence of length 25 to match the trained model's expected input. `maxlen=max_length-1 = 25` achieves this. If I used `maxlen=max_length=26`, the padded sequence would have 26 tokens — one more than the model expects — and TensorFlow would raise a shape error.

**Weak answer:** I pad to max_length minus one because the last element is the target.

**Common trap:** Not connecting the padding length at inference to the exact Input shape at training.

**Follow-up question:** If you loaded max_length from max_length.txt and forgot the -1, what would happen?

**Follow-up answer:** The inference would fail with a shape mismatch error: the model expects (None, 25) but would receive (1, 26). This is why the -1 must be applied consistently.

---

### HQ7. If a sentence in your dataset is longer than max_length words, what happens?

**What the interviewer is testing:** Understanding of sequence truncation in `pad_sequences`.

**Strong answer:** `max_length = max(lengths)` is computed from the actual dataset, so by definition no sentence is longer than max_length. The longest sentence in this dataset is 26 words, so max_length=26. If I were to add a 27-word sentence later, its full-length prefix `[t1, t2, …, t27]` would be truncated by `pad_sequences(maxlen=26, padding='pre')` to the last 26 tokens — losing the first token. This is `truncating='pre'` behavior (default). The sequence could also be truncated from the end if `truncating='post'` were specified.

**Weak answer:** It would get cut off.

**Common trap:** Not knowing which end gets truncated by default.

---

### HQ8. What is the output shape of the Embedding layer and why?

**What the interviewer is testing:** Understanding of how embedding layers transform input tensors.

**Strong answer:** Input to the Embedding is (batch_size, 25) — 25 integer indices per sample. The Embedding layer looks up each index in its weight matrix (654, 50) and replaces each index with its 50-dimensional embedding vector. The output is (batch_size, 25, 50) — for each of the 25 time steps, a 50-dimensional vector. The LSTM then processes this sequence of 25 vectors and outputs its final hidden state of size 50.

**Weak answer:** It outputs a vector for each word.

---

### HQ9. What does `mask_zero=True` actually propagate and where?

**What the interviewer is testing:** Understanding of Keras masking mechanics.

**Strong answer:** `mask_zero=True` causes the Embedding layer to compute a boolean mask tensor of shape (batch_size, 25). Positions where the input integer is 0 are marked as False (masked). Keras's Sequential API propagates this mask to the next layer — the LSTM — which uses it to skip hidden state updates for masked positions. This means positions 0–22 (the padding) for a 2-word input sequence do not update the hidden or cell state; only the positions containing real word indices do. The Dense layer receives the LSTM's final hidden state, which was only shaped by the real words.

**Weak answer:** It tells the model to ignore zeros.

---

### HQ10. How would you properly evaluate this model?

**What the interviewer is testing:** Experimental design and understanding of rigorous evaluation.

**Strong answer:**
1. Withhold a held-out test set of ~20 sentences *before any processing* — before fitting the tokenizer, before creating sequences.
2. Fit the tokenizer only on the remaining ~153 training sentences.
3. Map test sentences through the already-fitted tokenizer (some words may be OOV if the tokenizer has no OOV token — handle this).
4. Shuffle the training prefix sequences before splitting train/validation.
5. Use perplexity as the primary metric, not accuracy.
6. Evaluate the final model on the held-out test set exactly once, after all hyperparameter decisions are finalized.

**Weak answer:** Split into 80/20 and check val_accuracy.

---

### HQ11. The comment says `split(' ')` for computing lengths. Is this consistent with how the tokenizer works?

**What the interviewer is testing:** Attention to tokenization consistency.

**Strong answer:** The Keras Tokenizer by default applies `lower()` and filters punctuation before splitting. It does not simply split on spaces. `len(item.split(' '))` counts space-delimited tokens, which for clean lowercase sentences without punctuation should match the tokenizer's token count. However, if sentences contained punctuation, contractions, or multiple spaces, the two methods would disagree. For the actual 173 sentences in this dataset (clean, lowercase, no punctuation), the two methods produce the same lengths, so the max_length computation is practically correct — but it is technically inconsistent and could fail on different data.

---

### HQ12. The LSTM training uses `validation_split=0.2`. When exactly is this split applied?

**What the interviewer is testing:** Understanding of Keras internals.

**Strong answer:** Keras's `validation_split` takes the last `(1 - 0.8) × N` samples from the shuffled-or-unshuffled data **before** any batch iteration. The split is determined once at the start of `model.fit`. It does not shuffle — it literally takes `X[-343:]` and `Y[-343:]` as validation data and `X[:1374]` and `Y[:1374]` as training data. The model trains on training batches and evaluates on the full validation set at the end of each epoch.

---

## 36. Rapid-Fire Questions

**Q: What is an embedding?**
A dense trainable vector representation of a discrete token. Each word gets a unique row in a lookup table, updated by backpropagation.

**Q: What is an LSTM?**
A recurrent neural network with gating mechanisms (forget, input, output gates) and a cell state that allows learning over long sequences without vanishing gradients.

**Q: What is a hidden state?**
The LSTM's output at each time step — a vector summarizing the sequence seen so far. The final hidden state is passed to the Dense layer for prediction.

**Q: What is a cell state?**
The LSTM's internal memory, updated additively at each time step. It provides a direct gradient path for learning long-range dependencies.

**Q: What is masking?**
Telling the model to ignore certain time steps (padding positions). `mask_zero=True` creates a binary mask so the LSTM skips zero-padded positions.

**Q: What is padding?**
Adding zeros to shorter sequences so all sequences in a batch have the same length, which is required for parallel matrix operations.

**Q: Why `+1` in vocabulary size?**
To reserve index 0 for padding. Word indices run from 1 to 653; the embedding and output layer need 654 slots total.

**Q: What is `return_sequences`?**
A flag on LSTM that controls whether all time-step outputs are returned (True) or only the final hidden state (False). This project uses False.

**Q: What is dropout?**
Randomly zeroing a fraction of activations during training to prevent overfitting. Acts as an ensemble of smaller models.

**Q: What does Adam do?**
Maintains running estimates of gradient mean and variance per parameter, using them to adapt the learning rate individually for each weight.

**Q: Why softmax?**
To convert raw logits into a probability distribution over all vocabulary classes, required for categorical cross-entropy loss.

**Q: What is cross-entropy?**
A measure of how well the model's probability distribution matches the true distribution. For one-hot targets: $$-\log(\hat{y}_{\text{true}})$$.

**Q: What is an epoch?**
One full pass through the entire training dataset. This project trains for 100 epochs.

**Q: What is a batch?**
A subset of training examples processed together before one gradient update. This project uses batch_size=64.

**Q: What is validation split?**
Holding out a portion of training data to monitor performance on unseen data during training. This project uses 20%.

**Q: What is greedy decoding?**
Always selecting the highest-probability word at each inference step via argmax. Simple but produces repetitive output.

**Q: What is OOV?**
Out-of-vocabulary — a word not seen during tokenizer fitting. This project drops OOV words silently.

**Q: What is data leakage?**
When information from validation/test data influences model training, inflating performance estimates.

**Q: What is overfitting?**
When the model memorizes training data and fails to generalize. Evident here: training accuracy 36%, val accuracy 14.5% at epoch 100.

**Q: What is vanishing gradient?**
Gradients becoming exponentially small during backpropagation through many time steps, preventing learning of long-range dependencies. LSTMs mitigate this.

**Q: What is teacher forcing?**
Using the ground-truth previous word (not the model's prediction) as input at each training step. This happens implicitly in this project because training sequences are ground-truth prefixes. **Not explicitly designed as a technique.**

**Q: What is perplexity?**
$$e^{-\frac{1}{N}\sum \log P(w_i)}$$ — lower is better. The exponentiated average negative log-likelihood per word. A better metric than accuracy for language models. **Not computed in this project.**

**Q: How many training samples?**
Approximately 1,374 (80% of 1,717 total prefix sequences).

**Q: How many validation samples?**
Approximately 343 (20% of 1,717, taken from the tail of the ordered sequence list).

**Q: What is `fit_on_texts` vs `texts_to_sequences`?**
`fit_on_texts` builds the vocabulary from data (run once on training data). `texts_to_sequences` converts strings to integer lists using the built vocabulary (run on all data at inference too).

**Q: What is `word_counts`?**
A dictionary of word → raw frequency across all training sentences. Accessed via `tokenizer.word_counts`.

**Q: What is `document_count`?**
Total number of sentences (documents) seen by the tokenizer. Here: 173.

**Q: What is combinatorial explosion?**
When the number of combinations grows exponentially with the number of variables. Adding one more word list of size N multiplies the combination count by N.

**Q: What does `itertools.product` do?**
Returns the Cartesian product of input iterables — all possible ordered pairs (or tuples) of elements from each iterable.

**Q: What is `to_categorical`?**
Converts integer class labels to one-hot binary matrices. `to_categorical([2, 0, 1], 3)` → `[[0,0,1],[1,0,0],[0,1,0]]`.

**Q: What is `pad_sequences` default truncation?**
`truncating='pre'` — truncates from the left (keeps the tail of long sequences). Can be changed to 'post'.

**Q: What does `np.argmax` do?**
Returns the index of the maximum value in an array. Used in inference to find the predicted word index.

**Q: What is beam search?**
An inference algorithm that keeps the top-k most probable partial sequences at each step, exploring multiple paths instead of only the greedy-best. **Not implemented.**

**Q: What is the difference between `shuffle` and `sample`?**
`shuffle` permutes a list in-place (O(N)); `sample` returns a new list of k elements chosen without replacement (O(k)). For selecting 500 from 1,630, `sample` is more efficient.

**Q: Is this a classification or regression problem?**
Classification — predicting one of 654 discrete word classes.

**Q: What is sparse_categorical_crossentropy?**
Same as categorical_crossentropy but accepts integer labels (not one-hot vectors), saving memory and computation. Not used in this project.

**Q: What is recurrent_dropout?**
Dropout applied to the recurrent (h→h) connections in an LSTM. Separate from the input dropout. Not used in this project.

**Q: What format is the model saved in?**
`.keras` format — TensorFlow's native format. Saves architecture + weights + optimizer state.

**Q: Why pickle for the tokenizer?**
`pickle` serializes Python objects to binary. Keras Tokenizer has no built-in save method that preserves all attributes, so pickle is used. Security risk: only load pickled files from trusted sources.

---

## 37. Final Cheat Sheet

### Project Pipeline

```
173 hardcoded English sentences (in LSTM notebook)
        ↓
Tokenizer().fit_on_texts(dataset)
        ↓
653 unique words; word_index: {word → 1..653}
        ↓
Prefix sequence generation: 1,717 sequences of lengths 2..26
        ↓
pad_sequences(maxlen=26, padding='pre') → shape (1717, 26)
        ↓
X = [:, 0:-1] → (1717, 25)    Y = [:, -1] → (1717,) → one-hot (1717, 654)
        ↓
Embedding(654, 50, mask_zero=True)
        ↓
LSTM(50, dropout=0.5)
        ↓
Dense(654, softmax)
        ↓
fit(X, Y, epochs=100, batch_size=64, validation_split=0.2)
        ↓
Best val_accuracy: 0.1453 (epoch ~27–39)
        ↓
save model / tokenizer / max_length
        ↓
Inference: tokenize → pad(25) → argmax → index_word → append → repeat
```

### Most Important Formulas

**LSTM cell update:**
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

**Softmax:**
$$\hat{y}_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

**Categorical cross-entropy:**
$$\mathcal{L} = -\log(\hat{y}_{\text{true}})$$

**Adam update:**
$$\theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

**Perplexity (not computed, but know it):**
$$\text{PP} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log P(w_i)\right)$$

**LSTM parameter count:**
$$4 \times (d_{in} \times d_h + d_h^2 + d_h)$$

For this model: $$4 \times (50 \times 50 + 50 \times 50 + 50) = 20{,}200$$

**Embedding parameter count:**
$$(V+1) \times d_{emb} = 654 \times 50 = 32{,}700$$

**Dense parameter count:**
$$d_h \times (V+1) + (V+1) = 50 \times 654 + 654 = 33{,}354$$

### Most Important Interview Answers to Memorize

1. **Dataset:** 173 hardcoded sentences; 653 unique words; max_length=26.
2. **The file-writing bug:** `dataset.txt` contains 1,630 sentences, not 500; the truncation happens after the file write in execution order.
3. **The dataset generator was not used** in the final LSTM training.
4. **Sequence count:** 173 sentences → 1,717 prefix sequences.
5. **X shape:** (1717, 25); Y shape: (1717, 654) one-hot.
6. **Model:** Embedding(654, 50) → LSTM(50, dropout=0.5) → Dense(654, softmax). 86,254 total params.
7. **Why +1:** Index 0 is reserved for padding. 653 words + 1 = 654 total classes.
8. **Val accuracy:** 14.53% peak; training accuracy 36.3% at epoch 100. Severe overfitting.
9. **Inference is greedy decoding** (argmax) — produces degenerate repeating output.
10. **No OOV handling:** unknown words dropped silently.
11. **No true test set:** only training and (leaky) validation.
12. **Sequence-level leakage:** prefixes from same sentence in both train and val splits.
13. **Grid search mismatch:** final model uses embedding=50, LSTM=50 — not in the swept ranges.
14. **Early stopping exists but is commented out.**
15. **Inefficient word lookup:** O(V) linear scan instead of O(1) `index_word` lookup.
16. **The stale comment:** `# here we have 78 unique words` — actual vocab is 653.
17. **`mask_zero=True`:** causes LSTM to skip padded positions during hidden state updates.
18. **Pre-padding:** zeros on the left so real tokens are at the tail — recency preserved for final hidden state.
19. **`validation_split=0.2`:** takes last 20% of sequences sequentially, not randomly.
20. **X/Y split:** `pad_sequences[:, 0:-1]` for X and `pad_sequences[:, -1]` for Y.

### Biggest Project Weaknesses (Honest Summary)

| Weakness | Severity |
|---|---|
| No true test set | High |
| Sequence-level leakage in validation split | High |
| Dataset generator not used — disconnected experiment | Medium |
| Severe overfitting (val_acc 14.5% vs train_acc 36%) | High |
| Degenerate inference output (repetitive words) | High |
| Grid search hyperparameters not used in final model | Medium |
| Early stopping not applied | Medium |
| No OOV handling | Medium |
| O(V) word lookup at inference | Low |
| Stale incorrect comment (78 vs 653 words) | Low |
| File-writing bug in dataset generator | Medium |
| No shuffle before validation split | High |

### Best Improvements (Ranked by Impact)

1. Add a pre-split held-out test set (most impactful, zero code complexity)
2. Shuffle X/Y before `model.fit` (one line of code, eliminates split bias)
3. Enable early stopping (uncomment three existing lines)
4. Fix OOV: add `oov_token='<OOV>'` to Tokenizer constructor
5. Use `tokenizer.index_word[pos]` instead of linear scan
6. Use `sparse_categorical_crossentropy` to avoid large one-hot Y matrix
7. Include 50 in the grid search to validate the final model
8. Add a `tf.random.set_seed()` call for full reproducibility
9. Wire the dataset generator output to the LSTM notebook
10. Consider pre-trained embeddings (GloVe 50d) to compensate for tiny dataset

