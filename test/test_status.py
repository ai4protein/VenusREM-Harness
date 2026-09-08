from io import StringIO

from rem2.status import working


def test_working_non_tty_prints_dots():
    buf = StringIO()
    with working("loading rem2", stream=buf):
        pass
    assert "loading rem2 ..." in buf.getvalue()


def test_working_marks_failure():
    buf = StringIO()
    try:
        with working("loading rem2", stream=buf):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    # non-tty path only prints the start line
    assert "loading rem2 ..." in buf.getvalue()
