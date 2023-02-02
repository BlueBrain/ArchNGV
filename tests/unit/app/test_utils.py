import contextlib
import tempfile

from archngv.app import utils as tested
from archngv.app.utils import write_yaml


@contextlib.contextmanager
def temp_yaml_file(data_dict):
    with tempfile.NamedTemporaryFile(suffix=".yml") as tfile:
        write_yaml(tfile.name, data_dict)
        yield tfile.name


def test_load_ngv_manifest():
    old_manifest = {"common": {"p1": 1, "p2": 2}}

    with temp_yaml_file(old_manifest) as yaml_file:
        res = tested.load_ngv_manifest(yaml_file)
        assert res["common"]["p1"] == 1

    new_manifest = {"ngv": {"common": {"p1": 1, "p2": 2}}}

    with temp_yaml_file(new_manifest) as yaml_file:
        res = tested.load_ngv_manifest(yaml_file)
        assert res["common"]["p1"] == 1
