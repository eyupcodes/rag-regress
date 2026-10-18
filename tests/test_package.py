import rag_regress


def test_version_is_exposed() -> None:
    assert rag_regress.__version__.startswith("0.1.0")
