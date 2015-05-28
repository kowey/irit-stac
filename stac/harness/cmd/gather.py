# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)

"""
gather features
"""

from __future__ import print_function
from os import path as fp
import os

from attelo.harness.util import call, force_symlink

from ..local import (TEST_CORPUS,
                     TRAINING_CORPUS,
                     LEX_DIR,
                     ANNOTATORS)
from ..util import (current_tmp, latest_tmp)

NAME = 'gather'


def config_argparser(psr):
    """
    Subcommand flags.

    You should create and pass in the subparser to which the flags
    are to be added.
    """
    psr.add_argument("--skip-training",
                     default=False, action="store_true",
                     help="only gather test data")
    psr.set_defaults(func=main)


def extract_features(corpus, output_dir,
                     vocab_path=None):
    """
    Run feature extraction for a particular corpus; and store the
    results in the output directory. Output file name will be
    computed from the corpus file name

    :type: corpus: filepath

    :param: vocab_path: vocabulary to load for feature extraction
    (needed if extracting test data; must ensure we have the same
    vocab in test as we'd have in training)
    """
    # TODO: perhaps we could just directly invoke the appropriate
    # educe module here instead of going through the command line?
    cmd = ["stac-learning", "extract",
           corpus,
           LEX_DIR,
           output_dir,
           "--anno", ANNOTATORS]
    if vocab_path is not None:
        cmd.extend(['--vocabulary', vocab_path])
    call(cmd)
    call(cmd + ["--single"])


def main(args):
    """
    Subcommand main.

    You shouldn't need to call this yourself if you're using
    `config_argparser`
    """
    if args.skip_training:
        tdir = latest_tmp()
    else:
        tdir = current_tmp()
        extract_features(TRAINING_CORPUS, tdir)
    if TEST_CORPUS is not None:
        train_path = fp.join(tdir, fp.basename(TRAINING_CORPUS))
        vocab_path = train_path + '.relations.sparse.vocab'
        extract_features(TEST_CORPUS, tdir,
                         vocab_path=vocab_path)
    with open(os.path.join(tdir, "versions-gather.txt"), "w") as stream:
        call(["pip", "freeze"], stdout=stream)
    if not args.skip_training:
        latest_dir = latest_tmp()
        force_symlink(fp.basename(tdir), latest_dir)
