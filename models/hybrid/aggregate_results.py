#!/usr/bin/env python3
"""
Aggregate Results Across Multiple Manuscripts
==============================================
This script parses test results from multiple manuscripts and computes
average metrics (Precision, Recall, F1, IoU) across all manuscripts.

Usage:
    python3 aggregate_results.py --results_dir ./Results/UDIADS_BIB_MS --manuscripts Latin2 Latin14396 Latin16746 Syr341
"""

import argparse
import os
import re
import sys
from pathlib import Path


def parse_log_file(log_path):
    """
    Parse a log file to extract mean metrics.
    
    Args:
        log_path (str): Path to log file
        
    Returns:
        dict: Dictionary with keys 'precision', 'recall', 'f1', 'iou' or None if not found
    """
    if not os.path.exists(log_path):
        return None
    
    metrics = {}
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Look for mean metrics section
    # Format: "Mean Precision: 0.xxxx"
    precision_match = re.search(r'Mean Precision:\s+([\d.]+)', content)
    recall_match = re.search(r'Mean Recall:\s+([\d.]+)', content)
    f1_match = re.search(r'Mean F1:\s+([\d.]+)', content)
    iou_match = re.search(r'Mean IoU:\s+([\d.]+)', content)
    
    if precision_match and recall_match and f1_match and iou_match:
        metrics['precision'] = float(precision_match.group(1))
        metrics['recall'] = float(recall_match.group(1))
        metrics['f1'] = float(f1_match.group(1))
        metrics['iou'] = float(iou_match.group(1))
        return metrics
    
    return None


def find_latest_output_file(manuscript_dir):
    """
    Find the latest .out file in the manuscript directory.
    
    Args:
        manuscript_dir (str): Path to manuscript directory
        
    Returns:
        str: Path to latest .out file or None
    """
    if not os.path.exists(manuscript_dir):
        return None
    
    # Look for .out files in the directory
    out_files = list(Path(manuscript_dir).glob('*.out'))
    
    if not out_files:
        return None
    
    # Return the most recently modified file
    latest_file = max(out_files, key=lambda p: p.stat().st_mtime)
    return str(latest_file)


def find_log_file(manuscript_dir):
    """
    Find the test log file in the manuscript directory.
    
    Args:
        manuscript_dir (str): Path to manuscript directory
        
    Returns:
        str: Path to log file or None
    """
    # Check for test_log directory
    test_log_dir = Path(manuscript_dir).parent.parent / 'test_log'
    
    if test_log_dir.exists():
        log_files = list(test_log_dir.glob('test_log_*/*.txt'))
        if log_files:
            latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
            return str(latest_log)
    
    # Check for log files in manuscript directory
    log_files = list(Path(manuscript_dir).glob('*.txt'))
    if log_files:
        latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
        return str(latest_log)
    
    return None


def aggregate_metrics(results_dir, manuscripts):
    """
    Aggregate metrics across multiple manuscripts.
    
    Args:
        results_dir (str): Base directory containing manuscript results
        manuscripts (list): List of manuscript names
        
    Returns:
        dict: Aggregated metrics
    """
    all_metrics = []
    manuscript_results = {}
    
    print("\n" + "="*80)
    print("PARSING MANUSCRIPT RESULTS")
    print("="*80)
    
    for manuscript in manuscripts:
        manuscript_dir = os.path.join(results_dir, manuscript)
        
        print(f"\nProcessing: {manuscript}")
        print(f"  Directory: {manuscript_dir}")
        
        # Try to find metrics in multiple locations
        metrics = None
        
        # 1. Check for .out files
        out_file = find_latest_output_file(manuscript_dir)
        if out_file:
            print(f"  Found output file: {os.path.basename(out_file)}")
            metrics = parse_log_file(out_file)
        
        # 2. Check for log files if .out didn't work
        if metrics is None:
            log_file = find_log_file(manuscript_dir)
            if log_file:
                print(f"  Found log file: {os.path.basename(log_file)}")
                metrics = parse_log_file(log_file)
        
        # 3. Check parent directory for combined output files
        if metrics is None:
            parent_out_files = list(Path(results_dir).glob(f'*{manuscript}*.out'))
            if parent_out_files:
                latest_out = max(parent_out_files, key=lambda p: p.stat().st_mtime)
                print(f"  Found output in parent: {latest_out.name}")
                metrics = parse_log_file(str(latest_out))
        
        if metrics:
            all_metrics.append(metrics)
            manuscript_results[manuscript] = metrics
            print("  Metrics extracted:")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall:    {metrics['recall']:.4f}")
            print(f"    F1:        {metrics['f1']:.4f}")
            print(f"    IoU:       {metrics['iou']:.4f}")
        else:
            print(f"  Could not find metrics for {manuscript}")
            print("    Please ensure testing has completed and output files exist")
    
    if not all_metrics:
        print("\n" + "="*80)
        print("ERROR: No metrics found!")
        print("="*80)
        print("Please ensure that:")
        print("  1. Testing has completed for all manuscripts")
        print("  2. Output files (.out or .txt) exist in the results directory")
        print("  3. The results_dir path is correct")
        print("="*80)
        return None
    
    # Compute averages
    avg_precision = sum(m['precision'] for m in all_metrics) / len(all_metrics)
    avg_recall = sum(m['recall'] for m in all_metrics) / len(all_metrics)
    avg_f1 = sum(m['f1'] for m in all_metrics) / len(all_metrics)
    avg_iou = sum(m['iou'] for m in all_metrics) / len(all_metrics)
    
    return {
        'manuscripts': manuscript_results,
        'average': {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_f1,
            'iou': avg_iou
        },
        'num_manuscripts': len(all_metrics)
    }


def print_aggregated_results(results):
    """
    Print aggregated results in a nice format.
    
    Args:
        results (dict): Aggregated results dictionary
    """
    if results is None:
        return
    
    print("\n" + "="*80)
    print("INDIVIDUAL MANUSCRIPT RESULTS")
    print("="*80)
    print(f"{'Manuscript':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'IoU':<12}")
    print("-"*80)
    
    for manuscript, metrics in results['manuscripts'].items():
        print(f"{manuscript:<20} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
              f"{metrics['f1']:<12.4f} {metrics['iou']:<12.4f}")
    
    print("\n" + "="*80)
    print(f"AVERAGE METRICS ACROSS {results['num_manuscripts']} MANUSCRIPTS")
    print("="*80)
    avg = results['average']
    print(f"Average Precision: {avg['precision']:.4f}")
    print(f"Average Recall:    {avg['recall']:.4f}")
    print(f"Average F1 Score:  {avg['f1']:.4f}")
    print(f"Average IoU:       {avg['iou']:.4f}")
    print("="*80)


def save_aggregated_results(results, output_file):
    """
    Save aggregated results to a file.
    
    Args:
        results (dict): Aggregated results dictionary
        output_file (str): Path to output file
    """
    if results is None:
        return
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("AGGREGATED RESULTS ACROSS MANUSCRIPTS\n")
        f.write("="*80 + "\n\n")
        
        f.write("INDIVIDUAL MANUSCRIPT RESULTS\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Manuscript':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'IoU':<12}\n")
        f.write("-"*80 + "\n")
        
        for manuscript, metrics in results['manuscripts'].items():
            f.write(f"{manuscript:<20} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
                   f"{metrics['f1']:<12.4f} {metrics['iou']:<12.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write(f"AVERAGE METRICS ACROSS {results['num_manuscripts']} MANUSCRIPTS\n")
        f.write("="*80 + "\n")
        
        avg = results['average']
        f.write(f"Average Precision: {avg['precision']:.4f}\n")
        f.write(f"Average Recall:    {avg['recall']:.4f}\n")
        f.write(f"Average F1 Score:  {avg['f1']:.4f}\n")
        f.write(f"Average IoU:       {avg['iou']:.4f}\n")
        f.write("="*80 + "\n")
    
    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate test results across multiple manuscripts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Aggregate results for UDIADS-BIB manuscripts
  python3 aggregate_results.py --results_dir ./Results/UDIADS_BIB_MS \\
                                --manuscripts Latin2 Latin14396 Latin16746 Syr341
  
  # Aggregate with custom output file
  python3 aggregate_results.py --results_dir ./Results/UDIADS_BIB_MS \\
                                --manuscripts Latin2 Latin14396 Latin16746 Syr341 \\
                                --output aggregated_metrics.txt
        """
    )
    
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Base directory containing manuscript results')
    parser.add_argument('--manuscripts', nargs='+', required=True,
                       help='List of manuscript names to aggregate')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file to save aggregated results (optional)')
    
    args = parser.parse_args()
    
    # Aggregate metrics
    results = aggregate_metrics(args.results_dir, args.manuscripts)
    
    if results is None:
        sys.exit(1)
    
    # Print results
    print_aggregated_results(results)
    
    # Save to file if requested
    if args.output:
        save_aggregated_results(results, args.output)
    else:
        # Default output file
        default_output = os.path.join(args.results_dir, 'aggregated_metrics.txt')
        save_aggregated_results(results, default_output)


if __name__ == '__main__':
    main()
