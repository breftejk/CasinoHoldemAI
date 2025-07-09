import xgboost as xgb
import pandas as pd
from typing import List, Tuple
from .features import FeatureExtractor
from .simulator import MonteCarloSimulator

class PokerAI:
    def __init__(self, model_path: str, iters: int = 2000):
        """Load XGBoost model and init simulator."""
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        self.sim = MonteCarloSimulator(iters)

    def predict(self, player: List[str], board: List[str]) -> Tuple[str, float, float]:
        """
        Run Monte Carlo, extract features, then:
        - If win_rate >= 0.7, force CALL
        - Else use model_prob >= 0.5
        Returns: (decision, model_prob, win_rate)
        """
        win_rate, tie_rate = self.sim.simulate(player, board)
        feats = FeatureExtractor.extract(player, board, win_rate, tie_rate)
        df = pd.DataFrame([feats])
        dmat = xgb.DMatrix(df)
        model_prob = self.model.predict(dmat)[0]

        # Override: if you have >70% equity, always CALL
        if win_rate >= 0.7:
            decision = 'CALL'
        else:
            decision = 'CALL' if model_prob >= 0.5 else 'FOLD'

        return decision, model_prob, win_rate