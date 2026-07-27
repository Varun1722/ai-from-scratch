import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch.nn.functional as F
    import torch

    return mo, torch


@app.cell
def _(torch):
    torch.__version__
    return


@app.cell
def _():
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    print("Total number of character:", len(raw_text))
    print(raw_text[:99])
    return (raw_text,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now our goal is to tokenize this 20,479-character short story into individual words and spe-
    cial characters that we can then turn into embeddings for LLM training. But for understanding lets try to see how this will work with small data
    """)
    return


@app.cell
def _():
    import re

    text = "hello, we are builiding a tokeniser in python"
    result = re.split(r'(\s)', text)
    print(result)
    return re, text


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let’s modify the regular expression splits on whitespaces (\s), commas, and peri-
    ods ([,.]):
    """)
    return


@app.cell
def _(re, text):
    result_1 = re.split(r'([,.]|\s)',text)
    print(result_1)
    return (result_1,)


@app.cell
def _(result_1):
    #we can remove the whitespace characters 

    result_without_whitespace = [item for item in result_1 if item.strip()]
    print(result_without_whitespace)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The tokenization scheme we devised here works well on the simple sample text. Let’s
    modify it a bit further so that it can also handle other types of punctuation, such as ques-
    tion marks, quotation marks, and the double-dashes we have seen earlier in the first 100
    characters of Edith Wharton’s short story, along with additional special characters
    """)
    return


@app.cell
def _(re):
    text_1 = "Hello, world. Is this-- a test?"
    res = re.split(r'([,.:;?_!"()\']|--|\s)', text_1)
    res = [item for item in res if item.strip()]
    print(res)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now that we have a basic tokenizer working, let’s apply it to Edith Wharton’s entire
    short story
    """)
    return


@app.cell
def _(raw_text, re):
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    print(len(preprocessed))
    return (preprocessed,)


@app.cell
def _(preprocessed):
    print(preprocessed)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Converting tokens into token IDs

    we need to convert the python strings to integers to produce the token IDs.
    To map the previously generated tokens into token IDs, we have to build a vocabu-
    lary first. This vocabulary defines how we map each unique word and special character
    to a unique integer
    """)
    return


@app.cell
def _(preprocessed):
    all_words = sorted(set(preprocessed))
    vocab_size = len(all_words)
    print(vocab_size)
    return (all_words,)


@app.cell
def _(all_words):
    #Create a vocabulary
    vocab = {token:integer for integer, token in enumerate(all_words)}
    for i, item in enumerate(vocab.items()):
        print(item)
        if i>=50:
            break
    return (vocab,)


@app.cell
def _(re):
    #now lets try to implement a simple text tokenizer

    class SimpleTokenizerV1:
        def __init__(self, vocab):
            self.str_to_int = vocab
            self.int_to_str = {i:j for j,i in vocab.items()}

        def encode(self, text):
            preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
            preprocessed = [
                item.strip() for item in preprocessed if item.strip()
            ]
            ids = [self.str_to_int[s] for s in preprocessed]
            return ids

        def decode(self, ids):
            text = " ".join([self.int_to_str[i] for i in ids])
            text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
            return text

    return (SimpleTokenizerV1,)


@app.cell
def _(SimpleTokenizerV1, vocab):
    tokenizer = SimpleTokenizerV1(vocab)
    text_5 = """"It's the last he painted, you know,"
    Mrs. Gisburn said with pardonable pride."""
    ids = tokenizer.encode(text_5)
    print(ids)
    return ids, tokenizer


@app.cell
def _(ids, tokenizer):
    print(tokenizer.decode(ids))
    return


@app.cell
def _(tokenizer):
    #lets try to test for words not in the vocab i.e unknown words

    text_6 = "Hello, do you like tea?"
    print(tokenizer.encode(text_6))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We need to modify the tokenizer to handle unknown words. We also need to address
    the usage and addition of special context tokens that can enhance a model’s under-
    standing of context or other relevant information in the text. These special tokens
    can include markers for unknown words and document boundaries, for example. In
    particular, we will modify the vocabulary and tokenizer, SimpleTokenizerV2, to sup-
    port two new tokens, <|unk|> and <|endoftext|>
    """)
    return


@app.cell
def _(preprocessed):
    all_tokens = sorted(list(set(preprocessed)))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    vocab_new = {token:integer for integer,token in enumerate(all_tokens)}
    print(len(vocab_new.items()))
    return (vocab_new,)


@app.cell
def _(vocab_new):
    for idx, item_ in enumerate(list(vocab_new.items())[-5:]):
        print(item_)
    return


@app.cell
def _(re):
    class SimpleTokenizerV2:
        def __init__(self, vocab_new):
            self.str_to_int = vocab_new
            self.int_to_str = {i:s for s,i in vocab_new.items()}

        def encode(self, text):
            preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
            preprocessed = [
                item.strip() for item in preprocessed if item.strip()
            ]
            preprocessed = [item if item in self.str_to_int
                else "<|unk|>" for item in preprocessed]
            ids = [self.str_to_int[s] for s in preprocessed]
            return ids

        def decode(self, ids):
            text = " ".join([self.int_to_str[i] for i in ids])
            print(text)
            text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
            return text

    return (SimpleTokenizerV2,)


@app.cell
def _():
    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text9 = " <|endoftext|> ".join((text1, text2))
    print(text9)
    return (text9,)


@app.cell
def _(SimpleTokenizerV2, text9, vocab_new):
    tokenizer_new = SimpleTokenizerV2(vocab_new)
    print(tokenizer_new.encode(text9))

    return (tokenizer_new,)


@app.cell
def _(text9, tokenizer_new):
    tokenizer_new.decode(tokenizer_new.encode(text9))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    GPT models use a byte pair encoding tokenizer,
    which breaks words down into subword units
    """)
    return


@app.cell
def _():
    from importlib.metadata import version
    import tiktoken
    print("tiktoken version:", version("tiktoken"))
    return (tiktoken,)


@app.cell
def _(tiktoken):
    tokenizer_gpt = tiktoken.get_encoding("gpt2")
    return (tokenizer_gpt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The usage of this tokenizer is similar to the SimpleTokenizerV2 we implemented pre-
    viously via an encode method:
    """)
    return


@app.cell
def _(tokenizer_gpt):
    text10 = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."
    token_id_tiktoken = tokenizer_gpt.encode(text10, allowed_special={'<|endoftext|>'})

    print(token_id_tiktoken)
    return (token_id_tiktoken,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can then convert the token IDs back into text using the decode method, similar to
    our SimpleTokenizerV2:
    """)
    return


@app.cell
def _(token_id_tiktoken, tokenizer_gpt):
    strings_tokeniser_gpt = tokenizer_gpt.decode(token_id_tiktoken)
    print(strings_tokeniser_gpt)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Data Sampling with Sliding Window

    Let’s implement a data loader that fetches the input–target pairs from
    the training dataset using a sliding window approach.
    """)
    return


@app.cell
def _(raw_text, tokenizer_gpt):
    #We will tokenize the verdict using bpe tokenizer

    enc_text = tokenizer_gpt.encode(raw_text)
    print(len(enc_text))
    return (enc_text,)


@app.cell
def _(enc_text):
    enc_sample = enc_text[50:]
    print(enc_sample)
    return (enc_sample,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    One of the easiest and most intuitive ways to create the input–target pairs for the next-
    word prediction task is to create two variables, x and y, where x contains the input
    tokens and y contains the targets, which are the inputs shifted by 1
    """)
    return


@app.cell
def _(enc_sample):
    context_size = 4
    x = enc_sample[:context_size]
    y = enc_sample[1:context_size+1]
    print(f"x: {x}")
    print(f"y:     {y}")

    return (context_size,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    By processing the inputs along with the targets, which are the inputs shifted by one
    position, we can create the next-word prediction tasks
    """)
    return


@app.cell
def _(context_size, enc_sample, tokenizer_gpt):
    for idx_context in range(1, context_size+1):
        context = enc_sample[:idx_context]
        desired = enc_sample[idx_context]
        print(context, "---->", desired)
        print(tokenizer_gpt.decode(context), "---->", tokenizer_gpt.decode([desired]))
    return


@app.cell
def _(tokenizer_gpt, torch):
    from torch.utils.data import Dataset, DataLoader

    class GPTDatasetV1(Dataset):
        def __init__(self, txt,tokenizer,max_length, stride):
            self.input_ids = []
            self.target_ids = []

            token_ids = tokenizer_gpt.encode(txt)

            for i in range(0, len(token_ids)-max_length, stride):
                input_chunk=token_ids[i:i+max_length]
                target_chunk = token_ids[i+1:i+max_length+1]
                self.input_ids.append(torch.tensor(input_chunk))
                self.target_ids.append(torch.tensor(target_chunk))

        def __len__(self):
            return len(self.input_ids)
        def __getitem__(self, idx):
            return self.input_ids[idx], self.target_ids[idx]
            

    return DataLoader, GPTDatasetV1


@app.cell
def _(DataLoader, GPTDatasetV1, tiktoken):
    def create_dataloader_v1(txt, batch_size=4, max_length=256,
    stride=128, shuffle=True, drop_last=True,
    num_workers=0):
        tokenizer = tiktoken.get_encoding("gpt2")
        dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    
        dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    
        )

        return dataloader

    return (create_dataloader_v1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let’s test the dataloader with a batch size of 1 for an LLM with a context size of 4
    """)
    return


@app.cell
def _(create_dataloader_v1, raw_text):
    dataloader_sample = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)

    data_sample_iter = iter(dataloader_sample)
    first_sample_batch = next(data_sample_iter)
    print(first_sample_batch)
    return (data_sample_iter,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The first_batch variable contains two tensors: the first tensor stores the input token
    IDs, and the second tensor stores the target token IDs. Since the max_length is set to
    4, each of the two tensors contains four token IDs. Note that an input size of 4 is quite
    small and only chosen for simplicity.
    """)
    return


@app.cell
def _(data_sample_iter):
    #To understand the meaning of stride=1, let’s fetch another batch from this dataset:

    second_batch = next(data_sample_iter)
    print(second_batch)
    return


@app.cell
def _(create_dataloader_v1, raw_text):
    dataloader = create_dataloader_v1(
        raw_text, batch_size=8, max_length=4, stride=4,
        shuffle=False
    )

    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Inputs:\n", inputs)
    print("\nTargets:\n", targets)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Token Embeddings
    """)
    return


@app.cell
def _(torch):
    #example 

    sample_input_ids = torch.tensor([2,3,4,5])

    vocab_size_sample = 6
    output_dim_sample = 3

    torch.manual_seed(123)
    embedding_sample = torch.nn.Embedding(vocab_size_sample, output_dim_sample)
    print(embedding_sample.weight)
    return embedding_sample, sample_input_ids


@app.cell
def _(embedding_sample, torch):
    print(embedding_sample(torch.tensor([3])))
    return


@app.cell
def _(embedding_sample, sample_input_ids):
    print(embedding_sample(sample_input_ids))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Positional Encoding
    """)
    return


@app.cell
def _(torch):
    vocab_size_pos = 50257
    output_dim = 256
    token_embedding_layer = torch.nn.Embedding(vocab_size_pos, output_dim)
    return output_dim, token_embedding_layer


@app.cell
def _(create_dataloader_v1, raw_text):
    max_length = 4
    dataloader_pos = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
        stride=max_length, shuffle=False
    )
    data_iter_pos = iter(dataloader_pos)
    inputs_pos, targets_pos = next(data_iter_pos)
    print("Token IDs:\n", inputs_pos)
    print("\nInputs shape:\n", inputs_pos.shape)
    return inputs_pos, max_length


@app.cell
def _(inputs_pos, token_embedding_layer):
    token_embeddings = token_embedding_layer(inputs_pos)
    print(token_embeddings.shape)
    return (token_embeddings,)


@app.cell
def _(max_length, output_dim, torch):
    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    print(pos_embeddings.shape)
    return (pos_embeddings,)


@app.cell
def _(pos_embeddings, token_embeddings):
    input_embeddings = token_embeddings + pos_embeddings
    print(input_embeddings.shape)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
