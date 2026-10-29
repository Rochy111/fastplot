"""matplotlib.gridspec: GridSpec, SubplotSpec, GridSpecFromSubplotSpec."""
from ._layout import GridSpec, SubplotSpec


def GridSpecFromSubplotSpec(nrows, ncols, subplot_spec, wspace=None, hspace=None,  # noqa: N802
                            height_ratios=None, width_ratios=None):
    return GridSpec(nrows, ncols, parent=subplot_spec, wspace=wspace, hspace=hspace,
                    height_ratios=height_ratios, width_ratios=width_ratios)


__all__ = ['GridSpec', 'SubplotSpec', 'GridSpecFromSubplotSpec']
