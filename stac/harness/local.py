"""
Paths and settings used for this experimental harness
In the future we may move this to a proper configuration file.
"""

# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)

from __future__ import print_function
from os import path as fp
import itertools as itr

import educe.stac.corpus

from attelo.harness.config import (LearnerConfig,
                                   Keyed)
# from attelo.decoding.astar import (AstarArgs,
#                                    AstarDecoder,
#                                    Heuristic,
#                                    RfcConstraint)
from attelo.decoding.mst import (MstDecoder, MstRootStrategy)
from attelo.learning.local import (SklearnAttachClassifier,
                                   SklearnLabelClassifier)
from attelo.parser.intra import (IntraInterPair,
                                 HeadToHeadParser,
                                 # SentOnlyParser,
                                 SoftParser)
from attelo.util import (concat_l)

from sklearn.linear_model import (LogisticRegression)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


from .config.intra import (combine_intra)
# from .config.perceptron import (attach_learner_dp_pa,
#                                 attach_learner_dp_perc,
#                                 attach_learner_pa,
#                                 attach_learner_perc,
#                                 label_learner_dp_pa,
#                                 label_learner_dp_perc,
#                                 attach_learner_pa,
#                                 attach_learner_perc)
from .config.common import (ORACLE,
                            combined_key,
                            decoder_last,
                            decoder_local,
                            # mk_joint,
                            mk_post)

from .turn_constraint import (tc_decoder,
                              tc_learner)

# PATHS

CONFIG_FILE = fp.splitext(__file__)[0] + '.py'


LOCAL_TMP = 'TMP'
"""Things we may want to hold on to (eg. for weeks), but could
live with throwing away as needed"""

SNAPSHOTS = 'data/SNAPSHOTS'
"""Results over time we are making a point of saving"""


TRAINING_CORPUS = 'data/FROZEN/training-2015-05-30'
# TRAINING_CORPUS = 'data/tiny'
"""Corpora for use in building/training models and running our
incremental experiments. Later on we should consider using the
held-out test data for something, but let's make a point of
holding it out for now.

Note that by convention, corpora are identified by their basename.
Something like `data/socl-season1` would result
in a corpus named "socl-season1". This could be awkward if the basename
is used in more than one corpus, but we can revisit this scheme as
needed.
"""

# TEST_CORPUS = 'data/FROZEN/test-2015-05-30'
TEST_CORPUS = None
# TEST_CORPUS = 'tiny'
"""Corpora for use in FINAL testing.

You should probably leave this set to None until you've tuned and
tweaked things to the point of being able to write up a paper.
Wouldn't want to overfit to the test corpus, now would we?

(If you leave this set to None, we will perform 10-fold cross
validation on the training data)
"""

TEST_EVALUATION_KEY = None
# TEST_EVALUATION_KEY = 'maxent-AD.L_pst-tc-mst-root'
"""Evaluation to use for testing.

Leave this to None until you think it's OK to look at the test data.
The key should be the evaluation key from one of your EVALUATIONS,
eg. 'maxent-C0.9-AD.L_jnt-mst'

(HINT: you can join them together from the report headers)
"""

LEX_DIR = "lexicon"
"""
Lexicons used to help feature extraction
"""

ANNOTATORS = educe.stac.corpus.METAL_STR
"""
Which annotators to read from during feature extraction
"""

FIXED_FOLD_FILE = None
# FIXED_FOLD_FILE = 'folds-training-2015-05-30.json'
"""
Set this to a file path if you *always* want to use it for your corpus
folds. This is for long standing evaluation experiments; we want to
ensure that we have the same folds across different evaluate experiments,
and even across different runs of gather.

NB. It's up to you to ensure that the folds file makes sense
"""


DECODER_LOCAL = decoder_local(0.2)
"local decoder should accept above this score"


def decoder_mst():
    "our instantiation of the mst decoder"
    return Keyed('mst', MstDecoder(MstRootStrategy.fake_root, True))


def attach_learner_maxent():
    "return a keyed instance of maxent learner"
    return Keyed('maxent', SklearnAttachClassifier(LogisticRegression()))


def label_learner_maxent():
    "return a keyed instance of maxent learner"
    return Keyed('maxent', SklearnLabelClassifier(LogisticRegression()))


def attach_learner_dectree():
    "return a keyed instance of decision tree learner"
    return Keyed('dectree', SklearnAttachClassifier(DecisionTreeClassifier()))


def label_learner_dectree():
    "return a keyed instance of decision tree learner"
    return Keyed('dectree',
                 SklearnLabelClassifier(DecisionTreeClassifier()))


def attach_learner_rndforest():
    "return a keyed instance of random forest learner"
    return Keyed('rndforest',
                 SklearnAttachClassifier(RandomForestClassifier()))


def label_learner_rndforest():
    "return a keyed instance of decision tree learner"
    return Keyed('rndforest', SklearnLabelClassifier(RandomForestClassifier()))

_LOCAL_LEARNERS = [
    #    ORACLE,
    #    LearnerConfig(attach=attach_learner_maxent(),
    #                  label=label_learner_maxent()),
    LearnerConfig(attach=tc_learner(attach_learner_maxent()),
                  label=tc_learner(label_learner_maxent())),
    #    LearnerConfig(attach=attach_learner_maxent(),
    #                  label=label_learner_oracle()),
    #    LearnerConfig(attach=attach_learner_rndforest(),
    #                  label=label_learner_rndforest()),
    #    LearnerConfig(attach=attach_learner_perc(),
    #                  label=label_learner_maxent()),
    #    LearnerConfig(attach=attach_learner_pa(),
    #                  label=label_learner_maxent()),
    #    LearnerConfig(attach=attach_learner_dp_perc(),
    #                  label=label_learner_maxent()),
    #    LearnerConfig(attach=attach_learner_dp_pa(),
    #                  label=label_learner_maxent()),
]
"""Straightforward attelo learner algorithms to try

It's up to you to choose values for the key field that can distinguish
between different configurations of your learners.

"""


def _structured(klearner):
    """learner configuration pair for a structured learner

    (parameterised on a decoder)"""
    return lambda d: LearnerConfig(attach=tc_learner(klearner(d)),
                                   label=label_learner_maxent())


_STRUCTURED_LEARNERS = [
    #    _structured(attach_learner_dp_struct_perc),
    #    _structured(attach_learner_dp_struct_pa),
]
"""Attelo learners that take decoders as arguments.
We assume that they cannot be used relation modelling
"""


def _core_parsers(klearner):
    """Our basic parser configurations
    """
    # joint
    joint = [
        # mk_joint(klearner, decoder_last()),
        # mk_joint(klearner, DECODER_LOCAL),
        # mk_joint(klearner, decoder_mst()),
        # mk_joint(klearner, tc_decoder(DECODER_LOCAL)),
        # mk_joint(klearner, tc_decoder(decoder_mst())),
    ]

    # postlabeling
    post = [
        mk_post(klearner, decoder_last()),
        mk_post(klearner, DECODER_LOCAL),
        # mk_post(klearner, decoder_mst()),
        # mk_post(klearner, tc_decoder(DECODER_LOCAL)),
        mk_post(klearner, tc_decoder(decoder_mst())),
    ]
    if klearner.attach.payload.can_predict_proba:
        return joint + post
    else:
        return post

_INTRA_INTER_CONFIGS = [
    Keyed('iheads', HeadToHeadParser),
    # Keyed('ionly', SentOnlyParser),
    Keyed('isoft', SoftParser),
]


# -------------------------------------------------------------------------------
# maybe less to edit below but still worth having a glance
# -------------------------------------------------------------------------------

HARNESS_NAME = 'irit-stac'


def _mk_basic_intras(klearner, kconf):
    """Intra/inter parser based on a single core parser
    """
    return [combine_intra(IntraInterPair(x, x), kconf)
            for x in _core_parsers(klearner)]


def _mk_sorc_intras(klearner, kconf):
    """Intra/inter parsers based on a single core parser
    and a sentence oracle
    """
    parsers = [IntraInterPair(intra=x, inter=y) for x, y in
               zip(_core_parsers(ORACLE), _core_parsers(klearner))]
    return [combine_intra(p, kconf, primary='inter') for p in parsers]


def _mk_dorc_intras(klearner, kconf):
    """Intra/inter parsers based on a single core parser
    and a document oracle
    """
    parsers = [IntraInterPair(intra=x, inter=y) for x, y in
               zip(_core_parsers(klearner), _core_parsers(ORACLE))]
    return [combine_intra(p, kconf, primary='intra') for p in parsers]


def _mk_last_intras(klearner, kconf):
    """Intra/inter parsers based on a single core parser
    and the last baseline
    """
    kconf = Keyed(key=combined_key('last', kconf),
                  payload=kconf.payload)
    econf_last = mk_post(klearner, decoder_last())
    return [combine_intra(IntraInterPair(intra=econf_last, inter=p),
                          kconf,
                          primary='inter')
            for p in _core_parsers(klearner)]


def _is_junk(econf):
    """
    Any configuration for which this function returns True
    will be silently discarded
    """
    # intrasential head to head mode only works with mst for now
    has = econf.settings
    kids = econf.settings.children
    has_intra_oracle = has.intra and (kids.intra.oracle or kids.inter.oracle)
    has_any_oracle = has.oracle or has_intra_oracle

    decoder_name = econf.parser.key[len(has.key) + 1:]
    # last with last-based intra decoders is a bit redundant
    if has.intra and decoder_name == 'last':
        return True

    # oracle would be redundant with sentence/doc oracles
    if has.oracle and has_intra_oracle:
        return True

    # toggle or comment to enable filtering in/out oracles
    if has_any_oracle:
        return True

    return False


def _evaluations():
    "the evaluations we want to run"
    # non-prob mst decoder (dp learners don't do probs)
    nonprob_mst = Keyed('', MstDecoder(MstRootStrategy.fake_root, False))
    nonprob_mst = tc_decoder(nonprob_mst)
    nonprob_mst = nonprob_mst.payload
    #
    learners = []
    learners.extend(_LOCAL_LEARNERS)
    learners.extend(l(nonprob_mst) for l in _STRUCTURED_LEARNERS)
    ipairs = list(itr.product(learners, _INTRA_INTER_CONFIGS))
    res = concat_l([
        concat_l(_core_parsers(l) for l in learners),
        concat_l(_mk_basic_intras(l, x) for l, x in ipairs),
        concat_l(_mk_sorc_intras(l, x) for l, x in ipairs),
        concat_l(_mk_dorc_intras(l, x) for l, x in ipairs),
        concat_l(_mk_last_intras(l, x) for l, x in ipairs),
    ])
    return [x for x in res if not _is_junk(x)]


EVALUATIONS = _evaluations()


GRAPH_DOCS = [
    's2-league4-game1_07_stac_1396964826',
    's2-league4-game1_02_stac_1396964918',
]
"""Just the documents that you want to graph.
Set to None to graph everything
"""


def _want_details(econf):
    "true if we should do detailed reporting on this configuration"

    if isinstance(econf.learner, IntraInterPair):
        learners = [econf.learner.intra, econf.learner.inter]
    else:
        learners = [econf.learner]
    has_maxent = any('maxent' in l.key for l in learners)
    has = econf.settings
    kids = econf.settings.children
    has_intra_oracle = has.intra and (kids.intra.oracle or kids.inter.oracle)
    return (has_maxent and
            ('mst' in econf.parser.key or 'astar' in econf.parser.key) and
            not has_intra_oracle)

DETAILED_EVALUATIONS = [e for e in EVALUATIONS if _want_details(e)]
"""
Any evalutions that we'd like full reports and graphs for.
You could just set this to EVALUATIONS, but this sort of
thing (mostly the graphs) takes time and space to build

HINT: set to empty list for no graphs whatsoever
"""

# -------------------------------------------------------------------------------
# settings for the standalone parser
# -------------------------------------------------------------------------------


"""
The configuration we would like to use for the standalone parser.
"""

DIALOGUE_ACT_LEARNER = Keyed('maxent', LogisticRegression())
"""
Classifier to use for dialogue acts
"""

TAGGER_JAR = 'lib/ark-tweet-nlp-0.3.2.jar'
"POS tagger jar file"


CORENLP_DIR = 'lib/stanford-corenlp-full-2013-06-20'
"CoreNLP directory"


CORENLP_SERVER_DIR = "lib/corenlp-server"
"corenlp-server directory (see http://github.com/kowey/corenlp-server)"


CORENLP_ADDRESS = "tcp://localhost:5900"
"0mq address to server"

# -------------------------------------------------------------------------------
# nothing to edit below :-)
# -------------------------------------------------------------------------------


def print_evaluations():
    """
    Print out the name of each evaluation in our config
    """
    for econf in EVALUATIONS:
        print(econf)
        print()
    print("\n".join(econf.key for econf in EVALUATIONS))

if __name__ == '__main__':
    print_evaluations()
