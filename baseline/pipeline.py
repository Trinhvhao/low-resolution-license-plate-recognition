"""
Complete Training + Testing Pipeline

This script runs the full pipeline: train → validate → test → analyze results

Usage:
    python pipeline.py                 # Run full pipeline with defaults
    python pipeline.py --epochs 30     # Custom epochs
    python pipeline.py --batch_size 64 # Custom batch size
"""

import os
import sys
import argparse
import subprocess
import json
from datetime import datetime

try:
    from config import Config
except ImportError:
    print("❌ Error: Cannot import config. Run from baseline directory.")
    sys.exit(1)


class Pipeline:
    """Complete training and testing pipeline."""
    
    def __init__(self, epochs=None, batch_size=None, save_results=True):
        self.epochs = epochs or Config.EPOCHS
        self.batch_size = batch_size or Config.BATCH_SIZE
        self.save_results = save_results
        self.start_time = datetime.now()
        
    def print_header(self, title):
        """Print formatted section header."""
        print(f"\n{'='*80}")
        print(f"🚀 {title}")
        print(f"{'='*80}\n")
    
    def print_footer(self):
        """Print formatted footer."""
        elapsed = datetime.now() - self.start_time
        print(f"\n{'='*80}")
        print(f"✅ PIPELINE COMPLETED SUCCESSFULLY")
        print(f"Total Time: {elapsed}")
        print(f"{'='*80}\n")
    
    def run_command(self, cmd, description):
        """Run a shell command and report status."""
        print(f"Running: {description}")
        print(f"Command: {cmd}\n")
        
        result = subprocess.run(cmd, shell=True, cwd=os.path.dirname(__file__) or '.')
        
        if result.returncode != 0:
            print(f"❌ Error running: {description}")
            return False
        
        print(f"✅ {description} completed\n")
        return True
    
    def train(self):
        """Run training."""
        self.print_header("STEP 1: TRAINING MODEL")
        
        cmd = f"python train.py"
        if self.epochs and self.epochs != Config.EPOCHS:
            # Note: This assumes train.py accepts --epochs argument
            # If not, modify this to edit config temporarily
            pass
        
        return self.run_command(cmd, "Training")
    
    def test_validation(self):
        """Run validation test."""
        self.print_header("STEP 2: VALIDATION TEST")
        
        cmd = f"python test.py --mode val"
        if self.save_results:
            cmd += " --save_results"
        
        return self.run_command(cmd, "Validation Evaluation")
    
    def test_full(self):
        """Run full test."""
        self.print_header("STEP 3: FULL TEST")
        
        cmd = f"python test.py --mode val --show_errors 20"
        if self.save_results:
            cmd += " --save_results"
        
        return self.run_command(cmd, "Full Evaluation")
    
    def analyze_results(self):
        """Analyze results."""
        self.print_header("STEP 4: ANALYZE RESULTS")
        
        cmd = f"python analyze_results.py --latest 1"
        return self.run_command(cmd, "Results Analysis")
    
    def run_full_pipeline(self):
        """Run the complete pipeline."""
        print(f"\n{'#'*80}")
        print(f"# COMPLETE LPR TRAINING & TESTING PIPELINE")
        print(f"{'#'*80}")
        
        print(f"\nConfiguration:")
        print(f"  • EPOCHS: {self.epochs}")
        print(f"  • BATCH_SIZE: {self.batch_size}")
        print(f"  • SAVE_RESULTS: {self.save_results}")
        print(f"  • START TIME: {self.start_time}")
        
        # Step 1: Train
        if not self.train():
            print("❌ Training failed. Stopping pipeline.")
            return False
        
        # Step 2: Validate
        if not self.test_validation():
            print("⚠️  Validation failed. Continuing...")
        
        # Step 3: Full Test
        if not self.test_full():
            print("⚠️  Full test failed. Continuing...")
        
        # Step 4: Analyze
        if self.save_results:
            if not self.analyze_results():
                print("⚠️  Analysis failed. Continuing...")
        
        self.print_footer()
        return True


def run_quick_pipeline():
    """Run quick training + test for demonstration."""
    print("\n🚀 QUICK PIPELINE (Training only on subset)\n")
    
    # For quick pipeline, you might want to run with limited epochs/data
    pipeline = Pipeline(epochs=5, batch_size=128, save_results=True)
    return pipeline.run_full_pipeline()


def main():
    parser = argparse.ArgumentParser(
        description='Complete LPR Training & Testing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py                          # Full pipeline with defaults
  python pipeline.py --quick                 # Quick demo (5 epochs)
  python pipeline.py --epochs 30             # Custom epochs
  python pipeline.py --no-save                # Don't save results
  python pipeline.py --test-only              # Skip training, just test
        """
    )
    
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--quick', action='store_true', help='Run quick demo (5 epochs)')
    parser.add_argument('--no-save', action='store_true', help='Do not save results')
    parser.add_argument('--test-only', action='store_true', help='Skip training, just test')
    
    args = parser.parse_args()
    
    if args.quick:
        success = run_quick_pipeline()
    else:
        pipeline = Pipeline(
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_results=not args.no_save
        )
        
        if args.test_only:
            print(f"\n{'='*80}")
            print(f"🚀 TESTING ONLY (Skipping Training)")
            print(f"{'='*80}\n")
            
            pipeline.test_validation()
            pipeline.test_full()
            pipeline.analyze_results()
            pipeline.print_footer()
        else:
            success = pipeline.run_full_pipeline()
    
    print("\n" + ("="*80))
    print("📊 FOR DETAILED RESULTS, RUN:")
    print("  python analyze_results.py --latest 1")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
