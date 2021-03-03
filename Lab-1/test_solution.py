from lab1_solution import preProcess

def test_hello():
    assert True

def test_preProcess_wordonly_doNothing():
    str = "abcde"
    result = preProcess(str)

    assert str == result

def test_preProcess_2word_doNothing():
    str = "abcde abd"
    result = preProcess(str)

    assert str == result

def test_preProcess_wordSeperatedWithHashtag_deleteHashtag():
    str = "abcde#abd"
    expected = "abcdeabd"
    result = preProcess(str)

    assert expected == result