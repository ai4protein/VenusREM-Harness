from __future__ import annotations

from vrh.scoring.structure_weights import (
    classify_pdb_origin,
    load_residue_plddt_from_pdb,
    plddt_skip_reason,
)

_ATOM = (
    "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 {bf:5.2f}           C\n"
    "ATOM      2  CA  GLY A   2       1.000   0.000   0.000  1.00 {bf:5.2f}           C\n"
)


def _write(path, header: str, bfactor: float = 80.0):
    path.write_text(header + _ATOM.format(bf=bfactor))
    return str(path)


def test_xray_crystal_skips_plddt(tmp_path):
    pdb = _write(
        tmp_path / "xtal.pdb",
        "HEADER    HYDROLASE                               01-MAY-95   1ABC\n"
        "EXPDTA    X-RAY DIFFRACTION\n"
        "TITLE     CRYSTAL STRUCTURE OF A HYDROLASE\n",
        bfactor=25.0,
    )
    kind, detail = classify_pdb_origin(pdb)
    assert kind == "experimental"
    assert "X-RAY" in detail
    reason = plddt_skip_reason(pdb)
    assert reason is not None
    assert "no pLDDT" in reason
    assert load_residue_plddt_from_pdb(2, protein_name="xtal", pdb_file=pdb) is None


def test_nmr_also_skips_plddt(tmp_path):
    pdb = _write(
        tmp_path / "nmr.pdb",
        "EXPDTA    SOLUTION NMR\n",
        bfactor=0.0,
    )
    assert classify_pdb_origin(pdb)[0] == "experimental"
    assert plddt_skip_reason(pdb)


def test_alphafold_keeps_plddt(tmp_path):
    pdb = _write(
        tmp_path / "af.pdb",
        "TITLE     ALPHAFOLD MONOMER V2.0 PREDICTION FOR TEST\n"
        "REMARK 99 pLDDT SCORES STORED IN B-FACTOR COLUMN\n",
        bfactor=91.3,
    )
    assert classify_pdb_origin(pdb)[0] == "predicted"
    assert plddt_skip_reason(pdb) is None
    weights = load_residue_plddt_from_pdb(2, protein_name="af", pdb_file=pdb)
    assert weights is not None
    assert abs(float(weights[0, 0]) - 0.913) < 1e-3


def test_bfactors_over_100_treated_as_temperature(tmp_path):
    pdb = _write(tmp_path / "hot.pdb", "", bfactor=142.0)
    kind, detail = classify_pdb_origin(pdb)
    assert kind == "experimental"
    assert "B-factors > 100" in detail
    assert plddt_skip_reason(pdb)


def test_headerless_af_like_bfactors_allowed(tmp_path):
    pdb = _write(tmp_path / "plain.pdb", "", bfactor=72.0)
    assert classify_pdb_origin(pdb)[0] == "unknown"
    assert plddt_skip_reason(pdb) is None
    assert load_residue_plddt_from_pdb(2, protein_name="plain", pdb_file=pdb) is not None


def test_cli_warns_crystal_when_plddt_on(tmp_path):
    from types import SimpleNamespace

    from vrh.cli import _crystal_plddt_skip_paths, _warn_crystal_no_plddt

    xtal = _write(
        tmp_path / "xtal.pdb",
        "EXPDTA    X-RAY DIFFRACTION\n",
        bfactor=20.0,
    )

    class _Log:
        def __init__(self):
            self.msgs = []

        def warn(self, msg):
            self.msgs.append(msg)

    on = SimpleNamespace(use_plddt_decay=True, plddt_explicit=False, pdb=xtal, pdb_dir=None)
    assert _crystal_plddt_skip_paths(on)
    log = _Log()
    assert _warn_crystal_no_plddt(on, log) is True
    assert "Full vrh includes pLDDT" in log.msgs[0]
    assert "no pLDDT" in log.msgs[0]

    off = SimpleNamespace(use_plddt_decay=False, plddt_explicit=False, pdb=xtal, pdb_dir=None)
    assert _crystal_plddt_skip_paths(off) == []
    assert _warn_crystal_no_plddt(off, _Log()) is False

    explicit = SimpleNamespace(use_plddt_decay=True, plddt_explicit=True, pdb=xtal, pdb_dir=None)
    log2 = _Log()
    assert _warn_crystal_no_plddt(explicit, log2) is True
    assert "You enabled pLDDT" in log2.msgs[0]
