#from data_objects.utterance import Utterance
from utterance import Utterance
from pathlib import Path


# Contains the set of utterances of a single speaker
class Speaker:
    def __init__(self, root: Path, partition=None):
        self.root = root
        self.partition = partition
        self.name = root.name
        self.utterances = None
        self.utterance_cycler = None
        if self.partition is None:
            with self.root.joinpath("_source.txt").open("r") as sources_file:
                sources = [l.strip().split(",") for l in sources_file]
        else:
            with self.root.joinpath("sources_{}.txt".format(self.partition)).open("r") as sources_file:
                sources = [l.strip().split(",") for l in sources_file]
        self.sources = [[self.root, frames_fname, self.name, wav_path] for frames_fname, wav_path in sources]

    def _load_utterances(self):
        self.utterances = [Utterance(source[0].joinpath(source[1])) for source in self.sources]

    def random_partial(self, count, n_frames):
        """
        Samples a batch of <count> unique partial utterances from the disk in a way that all
        utterances come up at least once every two cycles and in a random order every time.

        :param count: The number of partial utterances to sample from the set of utterances from
        that speaker. Utterances are guaranteed not to be repeated if <count> is not larger than
        the number of utterances available.
        :param n_frames: The number of frames in the partial utterance.
        :return: A list of tuples (utterance, frames, range) where utterance is an Utterance,
        frames are the frames of the partial utterances and range is the range of the partial
        utterance with regard to the complete utterance.
        """
        if self.utterances is None:
            self._load_utterances()

        utterances = self.utterance_cycler.sample(count)
        print("N_frame in Speaker.py: ",n_frames)
        a = [(u,) + u.random_partial(n_frames) for u in utterances]

        return a