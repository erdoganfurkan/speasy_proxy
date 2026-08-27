import importlib
from types import SimpleNamespace

import pytest

m = importlib.import_module("speasy_proxy.api.v1.metadata")


def _inventory(monkeypatch, parameters=None, datasets=None):
    """Stands in for spz.inventories.flat_inventories.<provider>, no network, no real inventory."""
    provider = SimpleNamespace(parameters=parameters or {}, datasets=datasets or {})
    monkeypatch.setattr(m.spz, "inventories",
                        SimpleNamespace(flat_inventories=SimpleNamespace(archive=provider)),
                        raising=False)


def test_dims_of_a_scalar():
    assert m._dims(1) == (1, 1)


def test_dims_of_a_spectrogram():
    assert m._dims((96,)) == (96, 1)


def test_dims_past_two_are_dropped():
    # AMDA_Kernel's Pusher holds two dimensions at most (CONTAINER_MATRIX), so a
    # (11, 9, 96) parameter cannot be described here. Tracked with the DEPEND work.
    assert m._dims((11, 9, 96)) == (11, 9)


def test_dims_are_unknown_without_a_shape():
    assert m._dims(None) is None


@pytest.mark.parametrize("cdf_type,expected", [
    ("CDF_INT4", "INT"), ("CDF_UINT2", "INT"), ("CDF_BYTE", "INT"),
    ("CDF_DOUBLE", "DOUBLE"), ("CDF_REAL4", "DOUBLE"), (None, "DOUBLE"),
])
def test_amda_type(cdf_type, expected):
    assert m._amda_type(cdf_type) == expected


def test_a_complete_parameter_answers_all_four_keys(monkeypatch):
    _inventory(monkeypatch,
               parameters={"ddbase/ace_imf_all/IMF": SimpleNamespace(spz_shape=(3,), cdf_type="CDF_REAL4")},
               datasets={"ddbase/ace_imf_all": SimpleNamespace(MinSampling=16.0)})
    body = m._metadata_lines("archive/ddbase/ace_imf_all/IMF")
    assert body == "TYPE=DOUBLE\nDIM1=3\nDIM2=1\nMIN_SAMPLING=16.0\n"


def test_a_key_the_inventory_cannot_answer_is_omitted(monkeypatch):
    # AMDA_Kernel falls back on its own default for a key it does not find, so an
    # absent line is a better answer than a made up number.
    _inventory(monkeypatch,
               parameters={"ddbase/ds/P": SimpleNamespace(cdf_type="CDF_INT4")},
               datasets={"ddbase/ds": SimpleNamespace()})
    assert m._metadata_lines("archive/ddbase/ds/P") == "TYPE=INT\n"


def test_an_unknown_parameter_answers_nothing(monkeypatch):
    _inventory(monkeypatch, parameters={}, datasets={})
    assert m._metadata_lines("archive/ddbase/ds/nope") == ""


def test_an_unknown_provider_answers_nothing(monkeypatch):
    _inventory(monkeypatch)
    assert m._metadata_lines("nosuchprovider/ds/P") == ""


def test_a_path_without_a_dataset_answers_nothing(monkeypatch):
    _inventory(monkeypatch)
    assert m._metadata_lines("archive") == ""
    assert m._metadata_lines("archive/lonely") == ""


def test_the_route_is_served(tmp_path, monkeypatch):
    """The kernel reaches this over plain HTTP, so the route itself must be wired up."""
    monkeypatch.setenv("SPEASY_PROXY_CORE_INVENTORY_SHARED_PATH", str(tmp_path / "inv_shared"))
    from fastapi.testclient import TestClient
    from speasy_proxy import app

    with TestClient(app) as client:
        r = client.get("/metadata?path=archive/no/such/product")
        assert r.status_code == 404
        assert r.headers["content-type"].startswith("text/plain")
