from ..detail import preprocessing as _pdetail


def rewrite_vasculature_mesh(ngv_config, run_parallel):
    """ Load and write the vasculature using openmesh. Openmesh fixes
    issues with the vasculature, thus it ensures that no vertex numbering
    will happen afterwards.
    """
    from _pdetail.rewrite_vasculature_mesh import rewrite_vasculature
    rewrite_vasculature(ngv_config, None)
