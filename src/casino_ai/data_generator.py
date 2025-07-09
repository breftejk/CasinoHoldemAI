"""
Parallel data generation with outer parallelism and progress bar.
"""
import pandas as pd
from .simulator import MonteCarloSimulator
from .features import FeatureExtractor
from eval7 import Deck
from tqdm import tqdm
from joblib import Parallel, delayed
from typing import Dict

class DataGenerator:
    def __init__(self, iters=500, workers=8):
        self.sim = MonteCarloSimulator(iters)
        self.workers = workers

    def _gen_one(self, _: int) -> Dict:
        deck = Deck(); deck.shuffle()
        player = [str(deck.deal(1)[0]), str(deck.deal(1)[0])]
        board = [str(deck.deal(1)[0]) for _ in range(3)]
        win, tie = self.sim.simulate(player, board)
        feats = FeatureExtractor.extract(player, board, win, tie)
        feats['label'] = int(win >= 0.5)
        return feats

    def generate(self, n_samples: int) -> pd.DataFrame:
        results = Parallel(n_jobs=self.workers)(
            delayed(self._gen_one)(i) for i in tqdm(range(n_samples), desc='Generating')
        )
        return pd.DataFrame(results)
