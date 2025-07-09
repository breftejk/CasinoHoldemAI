"""
Command-line interface for CasinoHoldemAI.
"""
import argparse
import os
from casino_ai.data_generator import DataGenerator
from casino_ai.trainer import ModelTrainer
from casino_ai.ai import PokerAI


def main():
    parser = argparse.ArgumentParser('CasinoHoldemAI')
    sub = parser.add_subparsers(dest='cmd')

    gen = sub.add_parser('gen', help='Generate training data')
    gen.add_argument('--n', type=int, default=100000, help='Number of samples')
    gen.add_argument('--out', default='data/train.csv', help='Output CSV path')
    gen.add_argument('--iters', type=int, default=500, help='Monte Carlo iterations')
    gen.add_argument('--workers', type=int, default=8, help='Parallel workers')

    train = sub.add_parser('train', help='Train XGBoost model')
    train.add_argument('--in', dest='input', default='data/train.csv', help='Input CSV path')
    train.add_argument('--model', default='models/holdem.xgb', help='Model output path')

    pred = sub.add_parser('pred', help='Predict CALL/FOLD')
    pred.add_argument('--model', required=True, help='Trained model path')
    pred.add_argument('--cards', required=True, help='Player cards e.g. "10H,4C"')
    pred.add_argument('--board', required=True, help='Flop cards e.g. "AS,QH,2D"')
    pred.add_argument('--threshold', type=float, default=0.5, help='Probability threshold for CALL')

    args = parser.parse_args()
    if args.cmd == 'gen':
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        dg = DataGenerator(iters=args.iters, workers=args.workers)
        df = dg.generate(args.n)
        df.to_csv(args.out, index=False)
        print(f"Generated {len(df)} samples -> {args.out}")

    elif args.cmd == 'train':
        os.makedirs(os.path.dirname(args.model), exist_ok=True)
        mt = ModelTrainer()
        model = mt.train(args.input)
        mt.save(model, args.model)
        print(f"Model saved to {args.model}")


    elif args.cmd == 'pred':
        ai = PokerAI(args.model)
        player = args.cards.split(',')
        board = args.board.split(',')
        decision, model_prob, win_rate = ai.predict(player, board)
        final_decision = 'CALL' if model_prob >= args.threshold else 'FOLD'
        print(f"Decision: {final_decision} (model_prob: {model_prob:.2f}, win_rate: {win_rate:.2f}, threshold: {args.threshold})")

    else:
        parser.print_help()

if __name__ == '__main__':
    main()
