import json
from pathlib import Path

import pytest

from rag_regress.golden import GoldenSetError, load_golden_set


def write(tmp_path: Path, *rows: object, raw: str | None = None) -> Path:
    path = tmp_path / "golden.jsonl"
    text = raw if raw is not None else "\n".join(json.dumps(r) for r in rows)
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_valid_items_and_skips_blank_lines(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        raw=(
            '{"id": "q1", "query": "What is RAG?", "relevant_ids": ["d1", "d2"]}\n'
            "\n"
            '{"id": "q2", "query": "Ankara?", "relevant_ids": ["d3"], '
            '"reference_answer": "Capital of Türkiye", "tags": ["geo"]}\n'
        ),
    )

    items = load_golden_set(path)

    assert [i.id for i in items] == ["q1", "q2"]
    assert items[0].relevant_ids == ("d1", "d2")
    assert items[0].reference_answer is None
    assert items[1].tags == ("geo",)


def test_loads_file_with_utf8_bom(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    row = '{"id": "q1", "query": "Ankara?", "relevant_ids": ["d1"]}\n'
    path.write_bytes(b"\xef\xbb\xbf" + row.encode("utf-8"))

    assert [i.id for i in load_golden_set(path)] == ["q1"]


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({"query": "x", "relevant_ids": ["d"]}, "id: Field required"),
        ({"id": "q", "query": "  ", "relevant_ids": ["d"]}, "query must not be blank"),
        ({"id": "q", "query": "x", "relevant_ids": []}, "relevant_ids"),
        ({"id": "q", "query": "x", "relevant_ids": ["d", "d"]}, "repeat an id"),
        ({"id": "q", "query": "x", "relevant_ids": [" "]}, "blank ids"),
        ({"id": "q", "query": "x", "relevant_ids": ["d"], "extra": 1}, "extra"),
    ],
)
def test_rejects_invalid_items_with_line_number(
    tmp_path: Path, row: dict, message: str
) -> None:
    path = write(tmp_path, {"id": "ok", "query": "fine", "relevant_ids": ["d"]}, row)

    with pytest.raises(GoldenSetError, match=message) as exc:
        load_golden_set(path)

    assert f"{path}:2:" in str(exc.value)


def test_rejects_duplicate_ids_and_points_at_first(tmp_path: Path) -> None:
    row = {"id": "q1", "query": "x", "relevant_ids": ["d"]}
    path = write(tmp_path, row, row)

    with pytest.raises(GoldenSetError, match=r":2: duplicate id 'q1' .*line 1"):
        load_golden_set(path)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("{not json}\n", "invalid JSON"),
        ('["a list"]\n', "expected a JSON object"),
        ("\n\n", "golden set is empty"),
    ],
)
def test_rejects_malformed_files(tmp_path: Path, raw: str, message: str) -> None:
    with pytest.raises(GoldenSetError, match=message):
        load_golden_set(write(tmp_path, raw=raw))


def test_missing_file_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(GoldenSetError, match="file not found"):
        load_golden_set(tmp_path / "nope.jsonl")


def test_directory_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(GoldenSetError, match="cannot read"):
        load_golden_set(tmp_path)


def test_non_utf8_file_is_a_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_bytes('{"id": "q1"}'.encode("utf-16"))

    with pytest.raises(GoldenSetError, match="not valid UTF-8"):
        load_golden_set(path)


def test_unicode_line_separators_inside_strings_do_not_split_lines(
    tmp_path: Path,
) -> None:
    path = write(
        tmp_path,
        raw=(
            '{"id": "q1", "query": "first\u2028second\u0085third", '
            '"relevant_ids": ["d1"]}\r\n'
            '{"id": "q2", "query": "ok", "relevant_ids": ["d2"]}\r\n'
        ),
    )

    items = load_golden_set(path)

    assert [i.id for i in items] == ["q1", "q2"]
    assert items[0].query == "first\u2028second\u0085third"


def test_error_line_numbers_ignore_unicode_separators(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        raw=('{"id": "q1", "query": "a\u2028b", "relevant_ids": ["d1"]}\nnot json\n'),
    )

    with pytest.raises(GoldenSetError, match=r"golden\.jsonl:2: invalid JSON"):
        load_golden_set(path)
