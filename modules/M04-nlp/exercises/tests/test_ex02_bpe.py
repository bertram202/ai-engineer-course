from ex02_bpe import END, apply_bpe, merge_pair, pair_counts, train_bpe, word_to_symbols

# классический пример из статьи Sennrich и др. про BPE
CORPUS = {"low": 5, "lower": 2, "newest": 6, "widest": 3}


def test_pair_counts():
    counts = pair_counts({word_to_symbols(w): c for w, c in CORPUS.items()})
    assert counts[("e", "s")] == 9
    assert counts[("l", "o")] == 7
    assert counts[("w", "e")] == 8
    assert counts[("t", END)] == 9


def test_merge_pair():
    corpus = {("a", "a", "a", END): 2, ("b", "a", END): 1}
    assert merge_pair(corpus, ("a", "a")) == {("aa", "a", END): 2, ("b", "a", END): 1}


def test_merge_pair_combines_frequencies_of_equal_results():
    corpus = {("a", "b"): 1, ("ab",): 2}
    assert merge_pair(corpus, ("a", "b")) == {("ab",): 3}


def test_train_bpe_order_and_ties():
    merges = train_bpe(CORPUS, 5)
    # на первом шаге три пары с частотой 9: выигрывает наименьшая по алфавиту
    assert merges[:3] == [("e", "s"), ("es", "t"), ("est", END)]
    assert merges[3:] == [("l", "o"), ("lo", "w")]


def test_train_stops_when_nothing_to_merge():
    assert len(train_bpe({"ab": 1}, 10)) == 2  # a+b, затем ab+</w>


def test_apply_bpe():
    merges = train_bpe(CORPUS, 10)
    assert apply_bpe("lowest", merges) == ["low", "est</w>"]
    assert apply_bpe("newer", merges) == ["n", "ew", "e", "r", END]  # слияния r+</w> не выучено
    assert "".join(apply_bpe("widest", merges)) == "widest" + END
