# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)

"""show what evaluations we would run
"""

from __future__ import print_function

from ..harness import (IritHarness)

NAME = 'preview'


def config_argparser(psr):
    """
    Subcommand flags.

    You should create and pass in the subparser to which the flags
    are to be added.
    """
    psr.add_argument("--verbose",
                     default=False, action="store_true",
                     help="print details for all evaluations")
    psr.set_defaults(func=main)


def main(args):
    """
    Subcommand main.

    You shouldn't need to call this yourself if you're using
    `config_argparser`
    """
    hconf = IritHarness()
    if args.verbose:
        for econf in hconf.evaluations:
            # unfortunately we don't have a way of printing this nicely yet,
            # except maybe to override the __repr__ of EvaluationConfig,
            # which I'd be nervous about
            print(econf)
            print()
    print("Evaluation configs")
    print("------------------")
    print("\n".join(econf.key for econf in hconf.evaluations))
    print()
    print("TRAINING:", hconf.dataset)
    tconf = "(not enabled)" if hconf.test_evaluation is None\
        else "(config:{})".format(hconf.test_evaluation.key)
    print("TEST:", hconf.testset, tconf)
    if not args.verbose:
        print()
        print("Use --verbose for more details")
