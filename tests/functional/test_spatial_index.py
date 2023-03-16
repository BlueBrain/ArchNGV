from pathlib import Path

import spatial_index



BUILD_DIR = Path(__file__).parent.resolve() / "build"


def test_synapses():
    index_path = BUILD_DIR / 'spatial_index_synapses/meta_data.json'
    index = spatial_index.open_index(index_path.as_posix())
    assert len(index) > 0
    assert index.element_type == 'synapse'

