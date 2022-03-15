from pathlib import Path

import h5py
import libsonata
import numpy as np
import pytest
import yaml
from numpy import testing as npt

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
SPECS_DIR = ROOT_DIR / "specifications"
BUILD_DIR = Path(__file__).parent.resolve() / "build"


@pytest.fixture
def output_specs():
    with open(SPECS_DIR / "output.yaml", "r") as yaml_file:
        return yaml.load(yaml_file, Loader=yaml.FullLoader)


class SpecError:
    def __init__(self, msg=""):
        self.msg = msg

    def __str__(self):
        return self.msg


class H5SonataReader:
    """Sonata file hdf5 helper methods"""

    def __init__(self, filepath):
        self.h5 = h5py.File(filepath, "r")

    @property
    def _population_types_dict(self):
        return dict(self.h5.items())

    @property
    def population_types(self):
        return sorted(self._population_types_dict.keys())

    @property
    def _population_names_dict(self):
        return {
            pname: pgroup["0"]
            for tgroup in self._population_types_dict.values()
            for pname, pgroup in tgroup.items()
        }

    @property
    def population_names(self):
        return sorted(self._population_names_dict.keys())

    @property
    def attributes(self):
        return {
            key: group[key]
            for group in self._population_names_dict.values()
            for key in group
            if isinstance(group[key], h5py.Dataset)
        }

    @property
    def attribute_names(self):
        return sorted(self.attributes.keys())

    @property
    def attribute_dtypes(self):
        dtypes = {}
        for group in self._population_names_dict.values():

            keys = set(group.keys())
            if "@library" in keys:
                keys.remove("@library")
                for key, dataset in group["@library"].items():
                    dtypes[key] = dataset.dtype
                    keys.remove(key)

            for key in keys:
                dtypes[key] = group[key].dtype

        return dtypes

    @property
    def source_population_names(self):
        return sorted(
            pgroup["source_node_id"].attrs["node_population"]
            for pgroup in self.h5["edges"].values()
        )

    @property
    def target_population_names(self):
        return sorted(
            pgroup["target_node_id"].attrs["node_population"]
            for pgroup in self.h5["edges"].values()
        )


class SonataValidator:
    def __init__(self, build_dir, specification):

        self.build_dir = build_dir
        self.spec = specification

        self.error_list = []

    @property
    def filepath(self):
        return self.build_dir / Path(self.spec["filepath"])

    @property
    def h5file(self):
        assert Path(self.filepath).exists(), f"Filepath {self.filepath} does not exist."
        return H5SonataReader(self.filepath)

    def _collect_error(self, actual, desired, err_msg=""):
        try:
            npt.assert_equal(actual, desired)
        except AssertionError as e:

            flat_assertion_error = "\n".join(e.args)
            msg = f"\n{err_msg}{flat_assertion_error}\n"
            self.error_list.append(SpecError(msg))

    def check_population_name(self, spec):

        self._collect_error(
            actual=self.h5file.population_names,
            desired=[spec["population_name"]],
            err_msg=f"Population name mismatch in {self.filepath.name}",
        )

    def check_population_type(self, spec):

        self._collect_error(
            actual=self.h5file.population_types,
            desired=[spec["population_type"]],
            err_msg=f"Population type mismatch in {self.filepath.name}",
        )

    def check_attribute_names(self, spec):

        self._collect_error(
            actual=self.h5file.attribute_names,
            desired=sorted(spec["attributes"].keys()),
            err_msg=f"Mismatching attributes in {self.filepath.name}",
        )

    def check_attribute_dtypes(self, spec):

        dtypes = self.h5file.attribute_dtypes
        for attr_name, attr_spec in spec["attributes"].items():

            str_dtype = attr_spec["dtype"]

            if str_dtype == "string":
                str_dtype = "object"

            self._collect_error(
                actual=dtypes[attr_name],
                desired=np.dtype(str_dtype),
                err_msg=f"Dtype mismatch for '{attr_name}' in '{self.filepath.name}'",
            )

    def check_source_target_node_populations(self, spec):

        self._collect_error(
            actual=self.h5file.source_population_names,
            desired=[spec["source_node_population_name"]],
            err_msg=f"Source node population mismatch in {self.filepath.name}",
        )

        self._collect_error(
            actual=self.h5file.target_population_names,
            desired=[spec["target_node_population_name"]],
            err_msg=f"Target node population mismatch in {self.filepath.name}",
        )

    def run(self):

        self.check_population_type(self.spec)
        self.check_population_name(self.spec)
        self.check_attribute_names(self.spec)
        self.check_attribute_dtypes(self.spec)

        if self.spec["population_type"] == "edges":
            self.check_source_target_node_populations(self.spec)


def _validate_sonata_specs(specs):
    def assert_existence(key, dictionary, filepath, attr=None):

        if attr is None:
            msg = f"Missing '{key}' entry from the spec of '{filepath}'"
        else:
            msg = f"Missing '{key}' entry from attribute '{attr}' in the spec of '{filepath}'"

        assert key in dictionary, msg

    for n, spec in enumerate(specs):

        assert (
            "filepath" in spec
        ), f"Missing 'filepath' entry for the population at position {n} in the spec"

        filepath = spec["filepath"]
        assert_existence("description", spec, filepath)
        assert_existence("population_name", spec, filepath)
        assert_existence("population_type", spec, filepath)
        assert_existence("attributes", spec, filepath)

        if spec["population_type"] == "edges":
            assert_existence("source_node_population_name", spec, filepath)
            assert_existence("target_node_population_name", spec, filepath)

        for attr_name, attr_spec in spec["attributes"].items():
            assert_existence("description", attr_spec, filepath, attr=attr_name)
            assert_existence("dtype", attr_spec, filepath, attr=attr_name)


def test_sonata_specifications(output_specs):

    sonata_specs = output_specs["sonata"]
    _validate_sonata_specs(sonata_specs)

    errors = []

    for pop_dict in output_specs["sonata"]:

        validator = SonataValidator(BUILD_DIR, pop_dict)
        validator.run()

        errors.extend(validator.error_list)

    assert not errors, "\n".join(er.msg for er in errors)


def _check_dataset(filepath, dataset, layout):

    if dataset.ndim == 1:
        number_of_columns = 1
    else:
        number_of_columns = dataset.shape[1]

    npt.assert_equal(
        number_of_columns,
        int(layout["number_of_columns"]),
        err_msg=(f"Dataset {dataset.name} in {filepath} has inconsistent number of columns"),
    )

    npt.assert_equal(
        dataset.dtype,
        np.dtype(layout["dtype"]),
        err_msg=(f"Dataset {dataset.name} in {filepath} has inconsistent dtype"),
    )


def _check_hierarchy(filepath, h5file, hierarchy):

    group_keys = set(h5file)
    expected_group_keys = set(hierarchy)

    assert group_keys == expected_group_keys, (
        f"Mismmatch in {h5file.name} level:\n"
        f"Actual keys  : {group_keys}\n"
        f"Expected keys: {expected_group_keys}\n"
    )

    for name, sub_hierarchy in hierarchy.items():

        hdf5_object = h5file[name]
        hdf5_object_type = sub_hierarchy["object_type"]

        if hdf5_object_type == "group":

            assert isinstance(hdf5_object, h5py.Group)
            _check_hierarchy(filepath, hdf5_object, sub_hierarchy["contents"])

        else:

            assert isinstance(hdf5_object, h5py.Dataset)
            _check_dataset(filepath, hdf5_object, sub_hierarchy)


def test_custom_specifications(output_specs):

    for pop_dict in output_specs["custom"]:

        filepath = BUILD_DIR / Path(pop_dict["filepath"])
        assert filepath.exists(), f"Filepath {filepath} does not exist."

        with h5py.File(filepath, "r") as h5file:
            _check_hierarchy(filepath, h5file, pop_dict["file_hierarchy"])
