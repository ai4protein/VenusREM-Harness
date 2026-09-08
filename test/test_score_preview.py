import pandas as pd

from rem2.scoring.run_utils import (
    format_score_preview,
    has_experimental_dms,
    print_score_preview,
)


def test_format_score_preview_includes_dms_and_score():
    frame = pd.DataFrame(
        {
            "mutant": ["A16C", "A16D", "A16E"],
            "DMS_score": [-0.53, -2.15, -0.87],
            "esm2": [0.12, -0.4, 0.05],
        }
    )
    lines = format_score_preview(frame, "esm2", n=2)
    assert len(lines) == 3  # header + 2 rows
    assert "mutant" in lines[0]
    assert "DMS_score" in lines[0]
    assert "esm2" in lines[0]
    assert "A16C" in lines[1]
    assert "A16D" in lines[2]
    assert "A16E" not in "".join(lines)


def test_print_score_preview_logs_sample(capsys):
    class Logger:
        def info(self, message, protein=None):
            extra = f" [{protein}]" if protein else ""
            print(f"INFO{extra} {message}")

    frame = pd.DataFrame({"mutant": ["M1A"], "DMS_score": [1.0], "pred": [0.2]})
    print_score_preview(Logger(), frame, "pred", "demo_prot", n=5, path="out/scores/demo.csv")
    out = capsys.readouterr().out
    assert "Sample scores (1 of 1 mutants)" in out
    assert "demo_prot" in out
    assert "M1A" in out
    assert "out/scores/demo.csv" in out


def test_format_score_preview_accepts_raw_and_rem2_columns():
    frame = pd.DataFrame(
        {
            "mutant": ["A16C"],
            "DMS_score": [-0.5],
            "bb__raw_backbone": [0.1],
            "bb__rem2": [0.3],
        }
    )
    lines = format_score_preview(frame, ["bb__raw_backbone", "bb__rem2"], n=1)
    assert "bb__raw_backbone" in lines[0]
    assert "bb__rem2" in lines[0]


def test_has_experimental_dms():
    assert not has_experimental_dms(pd.DataFrame({"mutant": ["A1C"]}))
    assert not has_experimental_dms(pd.DataFrame({"mutant": ["A1C", "A1D"], "DMS_score": [0, 0]}))
    assert not has_experimental_dms(pd.DataFrame({"mutant": ["A1C", "A1D"]}))
    assert has_experimental_dms(
        pd.DataFrame({"mutant": ["A1C", "A1D"], "DMS_score": [-0.5, 1.2]})
    )
