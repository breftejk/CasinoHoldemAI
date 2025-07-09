"""
Monte Carlo simulation core.
"""
import eval7
from typing import List, Tuple

class MonteCarloSimulator:
    def __init__(self, iters: int = 500):
        self.iters = iters

    def _to_card(self, s: str) -> eval7.Card:
        """Convert string to eval7.Card: normalize rank/suit"""
        # Map '10' to 'T'
        raw_rank = s[:-1]
        rank = 'T' if raw_rank == '10' else raw_rank.upper()
        suit = s[-1].lower()
        return eval7.Card(rank + suit)

    def simulate(self, player: List[str], board: List[str]) -> Tuple[float, float]:
        """Run Monte Carlo and return (win_rate, tie_rate)."""
        wins = 0
        ties = 0
        # Normalize known cards
        known = [str(self._to_card(c)) for c in player + board]

        for _ in range(self.iters):
            deck = eval7.Deck()
            # Remove known cards
            deck.cards = [c for c in deck.cards if str(c) not in known]
            deck.shuffle()

            # Convert player and board to Card objects
            player_cards = [self._to_card(c) for c in player]
            board_cards  = [self._to_card(c) for c in board]

            # Deal dealer and extra cards
            dealer = [deck.deal(1)[0], deck.deal(1)[0]]
            extra  = [deck.deal(1)[0] for _ in range(2)]
            final_board = board_cards + extra

            # Evaluate
            pv = eval7.evaluate(player_cards + final_board)
            dv = eval7.evaluate(dealer + final_board)

            if pv > dv:
                wins += 1
            elif pv < dv:
                ties += 1

        return wins / self.iters, ties / self.iters