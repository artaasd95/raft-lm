"""
Compare multiple experiments and generate comparison report.

Usage:
    python scripts/compare_experiments.py --experiments exp1 exp2 exp3
"""

import argparse


def main():
    """
    Main comparison function.
    
    Loads results from multiple experiments and generates comparison.
    """
    parser = argparse.ArgumentParser(description='Compare multiple experiments')
    parser.add_argument(
        '--experiments',
        type=str,
        nargs='+',
        required=True,
        help='Names of experiments to compare'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='comparison_report.md',
        help='Output file for comparison report'
    )
    
    args = parser.parse_args()
    
    # TODO: Implement comparison pipeline
    # 1. Load results from each experiment
    # 2. Aggregate metrics across seeds
    # 3. Perform statistical tests
    # 4. Generate comparison tables
    # 5. Create visualization plots
    # 6. Write comparison report
    
    print(f"Comparing experiments: {args.experiments}")
    print("TODO: Implement comparison pipeline")


if __name__ == '__main__':
    main()

