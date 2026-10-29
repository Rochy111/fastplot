"""matplotlib.style: style.use('ggplot'), style.context(...), style.available
(matplotlib's bundled style sheets)."""
from ._rc import available as _available
from ._rc import context, use

available = _available()
library = {name: None for name in available}

__all__ = ['use', 'context', 'available', 'library']
