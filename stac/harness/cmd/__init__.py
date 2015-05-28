"""
irit-rst-dt subcommands
"""

# Author: Eric Kow
# License: CeCILL-B (French BSD3)

from . import (clean,
               count,
               evaluate,
               gather,
               model,
               parse,
               preview,
               serve,
               stop)

SUBCOMMANDS =\
    [
        gather,
        preview,
        evaluate,
        count,
        clean,
        model,
        parse,
        serve,
        stop,
    ]
