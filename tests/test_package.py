import rag_regress


def test_version_is_exposed() -> None:
    assert rag_regress.__version__.startswith("0.1.0")


def test_package_ships_py_typed_marker() -> None:
    """PEP 561: downstream type checkers only use our annotations with this marker."""
    from importlib import resources

    assert resources.files("rag_regress").joinpath("py.typed").is_file()
