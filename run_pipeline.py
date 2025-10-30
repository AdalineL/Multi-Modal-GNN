#!/usr/bin/env python
"""
EHR Graph Imputation Pipeline Runner

Interactive Python script to run the complete pipeline step-by-step.
Works on all platforms (Windows, macOS, Linux).

Usage:
    python run_pipeline.py

Or run individual steps:
    python run_pipeline.py --step 1    # Just preprocessing
    python run_pipeline.py --step 1-3  # Steps 1 through 3
"""

import sys
import subprocess
from pathlib import Path
import argparse


# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    """Print formatted header."""
    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{'='*70}{Colors.END}\n")


def print_success(text):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text):
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_info(text):
    """Print info message."""
    print(f"{Colors.YELLOW}→ {text}{Colors.END}")


# Pipeline steps
STEPS = [
    {
        'num': 1,
        'name': 'Preprocessing Data',
        'script': 'preprocess.py',
        'description': 'Load MIMIC-III data, extract features, normalize lab values'
    },
    {
        'num': 2,
        'name': 'Building Graph',
        'script': 'graph_build.py',
        'description': 'Construct heterogeneous graph with patients, labs, diagnoses, medications'
    },
    {
        'num': 3,
        'name': 'Visualizing Graph Structure',
        'script': 'visualize_graph.py',
        'description': 'Create graph structure visualizations (BEFORE training)'
    },
    {
        'num': 4,
        'name': 'Training Model',
        'script': 'train.py',
        'description': 'Train GNN with mask-and-recover strategy, early stopping'
    },
    {
        'num': 5,
        'name': 'Evaluating Model',
        'script': 'evaluate.py',
        'description': 'Compute metrics, baselines, per-lab performance'
    },
    {
        'num': 6,
        'name': 'Visualizing Results',
        'script': 'visualize.py',
        'description': 'Generate training curves, parity plots, error distributions'
    }
]


def run_step(step, src_dir, no_confirm=False):
    """Run a single pipeline step."""
    print_header(f"STEP {step['num']}: {step['name']}")
    print_info(f"Script: {step['script']}")
    print_info(f"Description: {step['description']}")
    print()

    # Ask for confirmation
    if not no_confirm:
        response = input(f"Press ENTER to run this step (or 's' to skip, 'q' to quit): ").strip().lower()

        if response == 'q':
            print_info("Pipeline cancelled by user")
            sys.exit(0)
        elif response == 's':
            print_info(f"Skipping step {step['num']}")
            return True

    # Run the script
    script_path = src_dir / step['script']

    try:
        print()
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=src_dir,
            check=True
        )

        print()
        print_success(f"Step {step['num']} completed successfully!")
        return True

    except subprocess.CalledProcessError as e:
        print()
        print_error(f"Step {step['num']} failed with error code {e.returncode}")

        if no_confirm:
            return False
        response = input("Continue with next step anyway? (y/N): ").strip().lower()
        return response == 'y'

    except FileNotFoundError:
        print_error(f"Script not found: {script_path}")
        return False


def main():
    """Main pipeline runner."""
    parser = argparse.ArgumentParser(description='Run EHR Graph Imputation Pipeline')
    parser.add_argument('--step', type=str, help='Run specific step(s) (e.g., "1" or "1-3")')
    parser.add_argument('--no-confirm', action='store_true', help='Run without confirmation prompts')
    args = parser.parse_args()

    # Get project directories
    project_dir = Path(__file__).parent
    src_dir = project_dir / 'src'

    print_header("EHR GRAPH IMPUTATION PIPELINE")
    print(f"Project directory: {project_dir}")
    print(f"Python version: {sys.version.split()[0]}")
    print()

    # Check if source directory exists
    if not src_dir.exists():
        print_error(f"Source directory not found: {src_dir}")
        sys.exit(1)

    # Determine which steps to run
    if args.step:
        if '-' in args.step:
            start, end = map(int, args.step.split('-'))
            steps_to_run = [s for s in STEPS if start <= s['num'] <= end]
        else:
            step_num = int(args.step)
            steps_to_run = [s for s in STEPS if s['num'] == step_num]
    else:
        steps_to_run = STEPS

    if not steps_to_run:
        print_error("No valid steps selected")
        sys.exit(1)

    print_info(f"Will run {len(steps_to_run)} step(s): {', '.join(str(s['num']) for s in steps_to_run)}")
    print()

    if not args.no_confirm:
        input("Press ENTER to start the pipeline...")

    # Run steps
    for step in steps_to_run:
        success = run_step(step, src_dir, args.no_confirm)
        if not success and not args.no_confirm:
            print_error("Pipeline stopped due to error")
            sys.exit(1)

    # Pipeline complete
    print_header("PIPELINE COMPLETE!")
    print("Results saved to:")
    print(f"  • {project_dir / 'data/interim'}         → Preprocessed data")
    print(f"  • {project_dir / 'outputs/graph.pt'}     → Saved graph")
    print(f"  • {project_dir / 'outputs/best_model.pt'} → Trained model")
    print(f"  • {project_dir / 'outputs/graph_visualizations/'} → Graph plots")
    print(f"  • {project_dir / 'outputs/visualizations/'}       → Results plots")
    print()


if __name__ == '__main__':
    main()
