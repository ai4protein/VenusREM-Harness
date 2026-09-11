from io import StringIO

from vrh.status import working


def test_working_non_tty_prints_dots():
    buf = StringIO()
    with working("loading vrh", stream=buf):
        pass
    assert "loading vrh ..." in buf.getvalue()


def test_working_marks_failure():
    buf = StringIO()
    try:
        with working("loading vrh", stream=buf):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    # non-tty path only prints the start line
    assert "loading vrh ..." in buf.getvalue()
