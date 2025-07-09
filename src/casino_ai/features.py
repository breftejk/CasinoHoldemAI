"""
Extract features for ML from card hands.
"""
from typing import List, Dict
from collections import Counter

class FeatureExtractor:
    RANK_ORDER = {
        '2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,
        'T':10,'10':10,'J':11,'Q':12,'K':13,'A':14
    }

    @staticmethod
    def extract(player: List[str], board: List[str], win_rate: float, tie_rate: float) -> Dict:
        """Return feature dict including all poker hand binary flags."""
        cards = player + board
        ranks, suits = [], []
        for c in cards:
            r, s = c[:-1], c[-1]
            ranks.append(r); suits.append(s)
        vals = [FeatureExtractor.RANK_ORDER[r] for r in ranks]
        val_counts = Counter(vals)
        suit_counts = Counter(suits)

        def is_flush(): return any(cnt >= 5 for cnt in suit_counts.values())
        def is_straight(vs):
            u = sorted(set(vs))
            for i in range(len(u)-4):
                if u[i+4] - u[i] == 4: return True
            if {14,2,3,4,5}.issubset(set(u)): return True
            return False

        has_pair = any(c >= 2 for c in val_counts.values())
        pairs = sum(1 for c in val_counts.values() if c == 2)
        has_trips = any(c >= 3 for c in val_counts.values())
        has_quads = any(c >= 4 for c in val_counts.values())
        has_full_house = has_trips and has_pair
        has_flush = is_flush()
        has_straight = is_straight(vals)
        has_sf = False
        for s in set(suits):
            suited_vals = [v for v, su in zip(vals, suits) if su == s]
            if is_straight(suited_vals): has_sf = True; break

        feats = {
            'win_rate': win_rate,
            'tie_rate': tie_rate,
            'pair': int(has_pair),
            'two_pair': int(pairs >= 2),
            'trips': int(has_trips),
            'quads': int(has_quads),
            'full_house': int(has_full_house),
            'flush': int(has_flush),
            'straight': int(has_straight),
            'straight_flush': int(has_sf)
        }
        for i, c in enumerate(player, 1):
            feats[f'rank_p{i}'] = FeatureExtractor.RANK_ORDER[c[:-1]]
            feats[f'suit_p{i}'] = ord(c[-1])
        for i, c in enumerate(board, 1):
            feats[f'board_r{i}'] = FeatureExtractor.RANK_ORDER[c[:-1]]
            feats[f'board_s{i}'] = ord(c[-1])
        return feats
