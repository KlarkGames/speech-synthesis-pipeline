import pytest

from src.datasets.EmoV_DB.preprocess import get_audio_id_to_text


def test_empty_string():
    cmuarctic_data = ""
    expected_result = {}
    assert get_audio_id_to_text(cmuarctic_data) == expected_result


def test_no_matches():
    cmuarctic_data = "This is a line with no matches"
    expected_result = {}
    assert get_audio_id_to_text(cmuarctic_data) == expected_result


def test_single_match():
    cmuarctic_data = '( arctic_a1234 "Hello World" )'
    expected_result = {1234: "Hello World"}
    assert get_audio_id_to_text(cmuarctic_data) == expected_result


def test_multiple_matches():
    cmuarctic_data = """
    ( arctic_a1234 "Hello World" )
    ( arctic_a5678 " Foo Bar" )
    """
    expected_result = {1234: "Hello World", 5678: " Foo Bar"}
    assert get_audio_id_to_text(cmuarctic_data) == expected_result


def test_arctic_bXXXX_ignored():
    cmuarctic_data = """
    ( arctic_a1234 "Hello World" )
    ( arctic_b5678 " Foo Bar" )
    """
    expected_result = {1234: "Hello World"}
    assert get_audio_id_to_text(cmuarctic_data) == expected_result


def test_audio_id_with_leading_zeros():
    cmuarctic_data = '( arctic_a01234 "Hello World" )'
    expected_result = {1234: "Hello World"}
    assert get_audio_id_to_text(cmuarctic_data) == expected_result


def test_text_with_quotes():
    cmuarctic_data = "( arctic_a1234 \"Hello 'World'\" )"
    expected_result = {1234: "Hello 'World'"}
    assert get_audio_id_to_text(cmuarctic_data) == expected_result


def test_text_with_escaped_quotes():
    cmuarctic_data = '( arctic_a1234 "Hello "World"" )'
    expected_result = {1234: 'Hello "World"'}
    assert get_audio_id_to_text(cmuarctic_data) == expected_result


@pytest.mark.parametrize(
    "cmuarctic_data, expected_result",
    [
        ("", {}),
        ("This is a line with no matches", {}),
        ('( arctic_a1234 "Hello World" )', {1234: "Hello World"}),
        (
            """
    ( arctic_a1234 "Hello World" )
    ( arctic_a5678 " Foo Bar" )
    """,
            {1234: "Hello World", 5678: " Foo Bar"},
        ),
        (
            """
    ( arctic_a1234 "Hello World" )
    ( arctic_b5678 " Foo Bar" )
    """,
            {1234: "Hello World"},
        ),
        ('( arctic_a01234 "Hello World" )', {1234: "Hello World"}),
        ("( arctic_a1234 \"Hello 'World'\" )", {1234: "Hello 'World'"}),
        ('( arctic_a1234 "Hello "World"" )', {1234: 'Hello "World"'}),
    ],
)
def test_get_audio_id_to_text(cmuarctic_data, expected_result):
    assert get_audio_id_to_text(cmuarctic_data) == expected_result
