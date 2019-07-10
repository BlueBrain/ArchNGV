from archngv.workflow.detail.preprocessing.rewrite_vasculature_mesh import rewrite_vasculature


def rewrite_vasculature_mesh(ngv_config, run_parallel):
    """ Load and write the vasculature using openmesh. Openmesh fixes
    issues with the vasculature, thus it ensures that no vertex numbering
    will happen afterwards.
    """
    rewrite_vasculature(ngv_config, None)
