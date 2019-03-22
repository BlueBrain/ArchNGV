import h5py


def extract_mesh_data(ngv_config, run_parallel):

    import trimesh
    mesh = trimesh.load(ngv_config.input_paths('vasculature_mesh'))

    path = ngv_config.output_paths('vasculature_mesh_data')

    with h5py.File(path, 'w') as fd:

        fd.create_dataset('coordinates', data=mesh.vertices)
        fd.create_dataset('faces', data=mesh.faces)

