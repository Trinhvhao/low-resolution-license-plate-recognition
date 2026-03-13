"""
Evaluation Results Analyzer - Compare and analyze test results

This script helps analyze and compare evaluation results from multiple test runs.

Usage:
    python analyze_results.py                              # List all results
    python analyze_results.py results1.json results2.json  # Compare two results
    python analyze_results.py --latest 3                  # Compare latest 3 results
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
import glob


class ResultAnalyzer:
    """Analyze and compare evaluation results."""
    
    def __init__(self):
        self.results = []
    
    def load_result(self, filepath):
        """Load result from JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                data['filepath'] = filepath
                self.results.append(data)
                return True
        except Exception as e:
            print(f"❌ Error loading {filepath}: {e}")
            return False
    
    def load_results_from_pattern(self, pattern):
        """Load all results matching a glob pattern."""
        files = sorted(glob.glob(pattern), reverse=True)
        count = 0
        for f in files:
            if self.load_result(f):
                count += 1
        return count
    
    def print_single_result(self, result):
        """Print a single result in formatted way."""
        print(f"\n{'='*80}")
        print(f"📊 EVALUATION RESULT")
        print(f"{'='*80}")
        
        if 'filepath' in result:
            print(f"File: {result['filepath']}")
        if 'timestamp' in result:
            print(f"Date: {result['timestamp']}")
        
        print(f"\nDataset: {result.get('dataset_name', 'Unknown')}")
        print(f"Total Samples: {result.get('total_samples', 'N/A')}")
        print(f"Correct Predictions: {result.get('correct_predictions', 'N/A')}")
        print(f"Accuracy: {result.get('accuracy', 'N/A'):.2f}%")
        print(f"Error Rate: {result.get('error_rate', 'N/A'):.2f}%")
        print(f"Average Loss: {result.get('avg_loss', 'N/A'):.4f}")
        
        if 'error_cases' in result and result['error_cases']:
            error_count = len(result['error_cases'])
            print(f"\nError Cases Recorded: {error_count}")
            if error_count > 0:
                print("Sample Errors:")
                for i, error in enumerate(result['error_cases'][:3]):
                    print(f"  {i+1}. Expected: '{error.get('ground_truth')}' | Got: '{error.get('prediction')}'")
                if error_count > 3:
                    print(f"  ... and {error_count - 3} more")
    
    def compare_results(self):
        """Compare multiple results."""
        if len(self.results) < 2:
            print("⚠️  Need at least 2 results to compare")
            return
        
        print(f"\n{'='*80}")
        print(f"📊 COMPARISON OF {len(self.results)} EVALUATION RUNS")
        print(f"{'='*80}\n")
        
        # Sort by timestamp
        sorted_results = sorted(
            self.results,
            key=lambda x: x.get('timestamp', ''),
            reverse=True
        )
        
        # Print table header
        print(f"{'#':<3} {'Timestamp':<20} {'Dataset':<10} {'Accuracy':<12} {'Samples':<10}")
        print(f"-" * 80)
        
        for i, result in enumerate(sorted_results):
            timestamp = result.get('timestamp', 'Unknown')[:19]
            dataset = result.get('dataset_name', 'Unknown')[:10]
            accuracy = f"{result.get('accuracy', 0):.2f}%"
            samples = result.get('total_samples', 0)
            
            print(f"{i+1:<3} {timestamp:<20} {dataset:<10} {accuracy:<12} {samples:<10}")
        
        # Print improvements
        print(f"\n{'='*80}")
        print(f"📈 IMPROVEMENTS:")
        print(f"{'='*80}\n")
        
        first_acc = sorted_results[-1].get('accuracy', 0)
        latest_acc = sorted_results[0].get('accuracy', 0)
        improvement = latest_acc - first_acc
        
        print(f"First Run Accuracy: {first_acc:.2f}%")
        print(f"Latest Run Accuracy: {latest_acc:.2f}%")
        print(f"Improvement: {improvement:+.2f}%")
        
        if improvement > 0:
            print(f"✅ Model improved by {improvement:.2f}%")
        elif improvement < 0:
            print(f"⚠️  Model degraded by {abs(improvement):.2f}%")
        else:
            print(f"➡️  No change in accuracy")
        
        # Detailed comparison
        if len(sorted_results) >= 2:
            print(f"\n{'='*80}")
            print(f"📋 DETAILED COMPARISON (Latest vs Previous):")
            print(f"{'='*80}\n")
            
            latest = sorted_results[0]
            previous = sorted_results[1]
            
            metrics = ['accuracy', 'error_rate', 'avg_loss', 'total_samples', 'correct_predictions']
            
            for metric in metrics:
                latest_val = latest.get(metric)
                previous_val = previous.get(metric)
                
                if latest_val is not None and previous_val is not None:
                    if metric == 'accuracy' or metric == 'error_rate':
                        change = latest_val - previous_val
                        print(f"{metric:<20} {previous_val:>8.2f}% → {latest_val:>8.2f}% ({change:+.2f}%)")
                    elif metric == 'avg_loss':
                        change = latest_val - previous_val
                        print(f"{metric:<20} {previous_val:>10.4f} → {latest_val:>10.4f} ({change:+.4f})")
                    else:
                        print(f"{metric:<20} {previous_val:>10} → {latest_val:>10}")
    
    def list_all_results(self):
        """List all available result files."""
        pattern = "outputs/evaluations/evaluation_results_*.json"
        files = sorted(glob.glob(pattern), reverse=True)
        
        if not files:
            print("❌ No evaluation result files found")
            return
        
        print(f"\n{'='*80}")
        print(f"📂 AVAILABLE EVALUATION RESULTS ({len(files)} files)")
        print(f"{'='*80}\n")
        
        for i, f in enumerate(files[:20], 1):
            try:
                with open(f, 'r') as file:
                    data = json.load(file)
                    timestamp = data.get('timestamp', 'Unknown')[:19]
                    accuracy = data.get('accuracy', 0)
                    dataset = data.get('dataset_name', 'Unknown')
                    
                    print(f"{i:2}. {f:<50} | {dataset:<5} | Acc: {accuracy:>6.2f}% | {timestamp}")
            except:
                print(f"{i:2}. {f:<50} | [Error reading file]")
        
        if len(files) > 20:
            print(f"\n... and {len(files) - 20} more files")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze and compare evaluation results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_results.py                    # List all results
  python analyze_results.py --latest 3         # Compare latest 3 results
  python analyze_results.py file1.json file2.json  # Compare specific files
        """
    )
    
    parser.add_argument('files', nargs='*', help='Result JSON files to analyze')
    parser.add_argument('--latest', type=int, help='Compare N latest results')
    parser.add_argument('--list', action='store_true', help='List all results')
    
    args = parser.parse_args()
    
    analyzer = ResultAnalyzer()
    
    # Load results
    if args.latest:
        # Load latest N results
        pattern = "outputs/evaluations/evaluation_results_*.json"
        files = sorted(glob.glob(pattern), reverse=True)[:args.latest]
        for f in files:
            analyzer.load_result(f)
        
        if analyzer.results:
            if len(analyzer.results) == 1:
                analyzer.print_single_result(analyzer.results[0])
            else:
                analyzer.compare_results()
        else:
            print("❌ No results found")
    
    elif args.files:
        # Load specific files
        for f in args.files:
            analyzer.load_result(f)
        
        if analyzer.results:
            if len(analyzer.results) == 1:
                analyzer.print_single_result(analyzer.results[0])
            else:
                analyzer.compare_results()
        else:
            print("❌ No results found")
    
    else:
        # List all results
        analyzer.list_all_results()
        
        print("\n📌 Usage Tips:")
        print("  • Compare latest 3 runs: python analyze_results.py --latest 3")
        print("  • Compare specific files: python analyze_results.py file1.json file2.json")


if __name__ == "__main__":
    main()
