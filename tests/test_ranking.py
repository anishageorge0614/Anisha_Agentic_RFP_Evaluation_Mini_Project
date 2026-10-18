import pytest

from rfp_eval.ranking import rank_suppliers, relative_percent

from .conftest import CRITERIA, make_supplier


def by_name(result):
    return {s["supplier_name"]: s for s in result["suppliers"]}


def test_formulas_hand_calculated():
    # A: tech 8, price 5 | B: tech 6, price 10  -> benchmarks: tech 8, price 10
    res = rank_suppliers([make_supplier("A", {1: 8, 2: 5}), make_supplier("B", {1: 6, 2: 10})], CRITERIA)
    a, b = by_name(res)["A"], by_name(res)["B"]
    assert res["benchmarks"] == {"1": 8.0, "2": 10.0}
    assert a["absolute_score"] == pytest.approx(8 / 10 * 60 + 5 / 10 * 40)  # 68
    assert b["absolute_score"] == pytest.approx(6 / 10 * 60 + 10 / 10 * 40)  # 76
    assert [c["gap"] for c in a["criteria"]] == [0.0, -5.0]
    assert [c["relative_pct"] for c in b["criteria"]] == [75.0, 100.0]
    assert a["ppi"] == pytest.approx(100 * 0.6 + 50 * 0.4)  # 80
    assert b["ppi"] == pytest.approx(75 * 0.6 + 100 * 0.4)  # 85
    assert [s["supplier_name"] for s in res["suppliers"]] == ["B", "A"]
    assert [s["final_rank"] for s in res["suppliers"]] == [1, 2]


def test_benchmark_leader_has_zero_gap_and_100_percent():
    res = rank_suppliers([make_supplier("A", {1: 9, 2: 9}), make_supplier("B", {1: 3, 2: 4})], CRITERIA)
    assert all(c["gap"] == 0 and c["relative_pct"] == 100 for c in by_name(res)["A"]["criteria"])
    assert all(c["gap"] < 0 for c in by_name(res)["B"]["criteria"])


def test_zero_benchmark_is_safe():
    assert relative_percent(0, 0) == 100.0
    res = rank_suppliers([make_supplier("A", {1: 0, 2: 5}), make_supplier("B", {1: 0, 2: 5})], CRITERIA)
    assert all(s["ppi"] == 100.0 for s in res["suppliers"])


def test_missing_score_excluded_from_benchmark_and_counted_zero():
    res = rank_suppliers([make_supplier("A", {1: None, 2: 5}), make_supplier("B", {1: 4, 2: 5})], CRITERIA)
    a = by_name(res)["A"]
    assert res["benchmarks"]["1"] == 4.0
    assert a["criteria"][0]["score"] == 0 and a["criteria"][0]["score_missing"]
    assert a["criteria"][0]["relative_pct"] == 0.0


def test_single_supplier():
    res = rank_suppliers([make_supplier("Solo", {1: 7, 2: 3})], CRITERIA)
    assert res["suppliers"][0]["ppi"] == 100.0 and res["suppliers"][0]["final_rank"] == 1


def test_weights_rescaled_when_not_100():
    crit = [{**CRITERIA[0], "weight": 30.0}, {**CRITERIA[1], "weight": 10.0}]
    res = rank_suppliers([make_supplier("A", {1: 10, 2: 10})], crit)
    assert res["suppliers"][0]["absolute_score"] == pytest.approx(100.0)
    assert res["original_weight_total"] == 40.0


EQUAL = {1: 7, 2: 7}


def order(*suppliers):
    res = rank_suppliers(list(suppliers), CRITERIA)
    return [s["supplier_name"] for s in res["suppliers"]], res


def test_tiebreak_1_higher_ppi():
    names, _ = order(make_supplier("Low", {1: 5, 2: 5}, "2026-01-01", 10), make_supplier("High", {1: 9, 2: 9}, "2026-12-01", 0))
    assert names == ["High", "Low"]


def test_tiebreak_2_earlier_submission_date():
    names, res = order(make_supplier("Late", EQUAL, "2026-09-05", 10), make_supplier("Early", EQUAL, "2026-09-01", 1))
    assert names == ["Early", "Late"]
    assert "earlier submission date" in res["suppliers"][1]["tie_break_note"]


def test_tiebreak_3_higher_experience():
    names, res = order(make_supplier("Junior", EQUAL, "2026-09-01", 4), make_supplier("Senior", EQUAL, "2026-09-01", 9))
    assert names == ["Senior", "Junior"]
    assert "experience" in res["suppliers"][1]["tie_break_note"]


def test_tiebreak_4_name_ascending():
    names, _ = order(make_supplier("Zeta", EQUAL), make_supplier("alpha", EQUAL), make_supplier("Mid", EQUAL))
    assert names == ["alpha", "Mid", "Zeta"]


def test_deterministic_regardless_of_input_order():
    s = [make_supplier(n, EQUAL) for n in "DBAC"]
    a = rank_suppliers(s, CRITERIA)
    b = rank_suppliers(list(reversed(s)), CRITERIA)
    assert [x["supplier_name"] for x in a["suppliers"]] == [x["supplier_name"] for x in b["suppliers"]] == list("ABCD")


def test_ranks_are_sequential():
    res = rank_suppliers([make_supplier(n, EQUAL) for n in "ABCDE"], CRITERIA)
    assert [s["final_rank"] for s in res["suppliers"]] == [1, 2, 3, 4, 5]
