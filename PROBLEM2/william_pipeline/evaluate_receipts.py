import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse

def parse_timestamp(ts):
    """Convert timestamp string to seconds"""
    if isinstance(ts, str):
        # Handle different timestamp formats
        try:
            if ":" in ts:
                # Format like "0:01:23.456"
                parts = ts.split(":")
                if len(parts) == 3:
                    hours, minutes, seconds = parts
                    seconds = float(seconds)
                    return int(hours) * 3600 + int(minutes) * 60 + seconds
                else:
                    minutes, seconds = parts
                    seconds = float(seconds)
                    return int(minutes) * 60 + seconds
            else:
                # Try parsing as float directly
                return float(ts)
        except:
            return 0
    return 0

def load_receipt(file_path):
    """Load a receipt file and standardize its format"""
    try:
        df = pd.read_csv(file_path)
        
        # Standardize column names (case insensitive)
        df.columns = [col.lower().strip() for col in df.columns]
        
        # Map common column variations to standard names
        name_mapping = {
            'product': 'product',
            'product_name': 'product',
            'item': 'product',
            'name': 'product',
            'timestamp': 'timestamp',
            'time': 'timestamp',
            'track_id': 'track_id',
            'trackid': 'track_id',
            'id': 'track_id'
        }
        
        # Rename columns based on mapping
        for col in df.columns:
            for key, value in name_mapping.items():
                if col == key:
                    df = df.rename(columns={col: value})
                    break
                    
        # Ensure required columns exist
        required_cols = ['timestamp', 'product']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            print(f"Error: Missing required columns {missing} in {file_path}")
            return None
            
        # Extract the main part of the product name (before the parentheses if present)
        if 'product' in df.columns:
            df['product_clean'] = df['product'].apply(
                lambda x: x.split('(')[0].strip() if isinstance(x, str) else x
            )
            
        # Parse timestamps to seconds
        if 'timestamp' in df.columns:
            df['timestamp_seconds'] = df['timestamp'].apply(parse_timestamp)
            
        return df
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return None

def compare_receipts(test_file, reference_file, time_tolerance=2.0):
    """
    Compare a test receipt to a reference receipt
    
    Args:
        test_file: Path to the test receipt CSV
        reference_file: Path to the reference receipt CSV
        time_tolerance: Tolerance in seconds for timestamp matching
        
    Returns:
        tuple: (matched_items, missed_items, extra_items, accuracy_metrics)
    """
    test_df = load_receipt(test_file)
    ref_df = load_receipt(reference_file)
    
    if test_df is None or ref_df is None:
        return [], ref_df['product_clean'].tolist() if ref_df is not None else [], [], {
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'match_rate': 0
        }
    
    # Match items between test and reference
    matched_items = []
    missed_items = []
    extra_items = []
    
    # Track used reference items to avoid duplicates
    used_ref_indices = set()
    
    # For each item in the test receipt
    for i, test_row in test_df.iterrows():
        test_product = test_row['product_clean']
        test_time = test_row['timestamp_seconds']
        
        best_match_idx = None
        min_time_diff = float('inf')
        
        # Find the best matching reference item
        for j, ref_row in ref_df.iterrows():
            if j in used_ref_indices:
                continue
                
            ref_product = ref_row['product_clean']
            ref_time = ref_row['timestamp_seconds']
            
            # Check if products match
            if test_product.lower() == ref_product.lower():
                time_diff = abs(test_time - ref_time)
                
                # If time difference is within tolerance and better than previous matches
                if time_diff <= time_tolerance and time_diff < min_time_diff:
                    min_time_diff = time_diff
                    best_match_idx = j
        
        # If a match was found
        if best_match_idx is not None:
            matched_items.append({
                'product': test_product,
                'test_time': test_row['timestamp'],
                'ref_time': ref_df.iloc[best_match_idx]['timestamp']
            })
            used_ref_indices.add(best_match_idx)
        else:
            # No match in reference - this is an extra item
            extra_items.append({
                'product': test_product,
                'test_time': test_row['timestamp']
            })
    
    # Find missed items (in reference but not matched)
    for j, ref_row in ref_df.iterrows():
        if j not in used_ref_indices:
            missed_items.append({
                'product': ref_row['product_clean'],
                'ref_time': ref_row['timestamp']
            })
    
    # Calculate metrics
    total_ref_items = len(ref_df)
    total_test_items = len(test_df)
    total_matched = len(matched_items)
    
    precision = total_matched / total_test_items if total_test_items > 0 else 0
    recall = total_matched / total_ref_items if total_ref_items > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    match_rate = total_matched / total_ref_items if total_ref_items > 0 else 0
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'match_rate': match_rate,
        'matched_count': total_matched,
        'reference_count': total_ref_items,
        'test_count': total_test_items,
        'missed_count': len(missed_items),
        'extra_count': len(extra_items)
    }
    
    return matched_items, missed_items, extra_items, metrics

def evaluate_receipts(test_dir, reference_dir, output_file=None, time_tolerance=2.0):
    """
    Evaluate all receipt files in the test directory against reference files
    
    Args:
        test_dir: Directory containing test receipt CSV files
        reference_dir: Directory containing reference receipt CSV files
        output_file: Optional path to write results to
        time_tolerance: Tolerance in seconds for timestamp matching
        
    Returns:
        dict: Summary metrics and detailed results
    """
    results = {}
    overall_metrics = {
        'total_matched': 0,
        'total_missed': 0,
        'total_extra': 0,
        'total_reference': 0,
        'total_test': 0,
        'avg_precision': 0,
        'avg_recall': 0,
        'avg_f1': 0,
    }
    
    # Find matching file names between test and reference directories
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.csv')]
    ref_files = [f for f in os.listdir(reference_dir) if f.endswith('.csv')]
    
    # Create mapping between test and reference files by their base names
    processed_files = set()
    for test_file in test_files:
        test_base = os.path.splitext(test_file)[0]
        
        # Try to find a matching reference file
        ref_file = None
        for rf in ref_files:
            ref_base = os.path.splitext(rf)[0]
            if test_base == ref_base or test_base in ref_base or ref_base in test_base:
                ref_file = rf
                break
        
        if ref_file:
            print(f"Comparing {test_file} with {ref_file}...")
            test_path = os.path.join(test_dir, test_file)
            ref_path = os.path.join(reference_dir, ref_file)
            
            matched, missed, extra, metrics = compare_receipts(test_path, ref_path, time_tolerance)
            
            # Store results
            results[test_file] = {
                'reference_file': ref_file,
                'matched_items': matched,
                'missed_items': missed,
                'extra_items': extra,
                'metrics': metrics
            }
            
            # Update overall metrics
            overall_metrics['total_matched'] += metrics['matched_count']
            overall_metrics['total_missed'] += metrics['missed_count']
            overall_metrics['total_extra'] += metrics['extra_count']
            overall_metrics['total_reference'] += metrics['reference_count']
            overall_metrics['total_test'] += metrics['test_count']
            
            processed_files.add(test_file)
            processed_files.add(ref_file)
        else:
            print(f"⚠️ No matching reference file found for {test_file}")
            
    # Check for reference files without matching test files
    for ref_file in ref_files:
        if ref_file not in processed_files:
            print(f"⚠️ No matching test file found for reference {ref_file}")
    
    # Calculate overall metrics
    if len(results) > 0:
        overall_metrics['avg_precision'] = sum(r['metrics']['precision'] for r in results.values()) / len(results)
        overall_metrics['avg_recall'] = sum(r['metrics']['recall'] for r in results.values()) / len(results)
        overall_metrics['avg_f1'] = sum(r['metrics']['f1_score'] for r in results.values()) / len(results)
        
        # Calculate overall precision and recall directly
        total_matched = overall_metrics['total_matched']
        total_ref = overall_metrics['total_reference']
        total_test = overall_metrics['total_test']
        
        overall_metrics['overall_precision'] = total_matched / total_test if total_test > 0 else 0
        overall_metrics['overall_recall'] = total_matched / total_ref if total_ref > 0 else 0
        overall_metrics['overall_f1'] = (
            2 * overall_metrics['overall_precision'] * overall_metrics['overall_recall'] / 
            (overall_metrics['overall_precision'] + overall_metrics['overall_recall'])
            if (overall_metrics['overall_precision'] + overall_metrics['overall_recall']) > 0 else 0
        )
    
    # Print summary
    print("\n===== EVALUATION SUMMARY =====")
    print(f"Files compared: {len(results)}")
    print(f"Overall precision: {overall_metrics['overall_precision']:.2%}")
    print(f"Overall recall: {overall_metrics['overall_recall']:.2%}")
    print(f"Overall F1 score: {overall_metrics['overall_f1']:.2%}")
    print(f"Total matched items: {overall_metrics['total_matched']}")
    print(f"Total reference items: {overall_metrics['total_reference']}")
    print(f"Total test items: {overall_metrics['total_test']}")
    print(f"Total missed items: {overall_metrics['total_missed']}")
    print(f"Total extra items: {overall_metrics['total_extra']}")
    
    # Generate detailed report
    if output_file:
        with open(output_file, 'w') as f:
            f.write("# Receipt Evaluation Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Overall Results\n\n")
            f.write(f"- Files compared: {len(results)}\n")
            f.write(f"- Overall precision: {overall_metrics['overall_precision']:.2%}\n")
            f.write(f"- Overall recall: {overall_metrics['overall_recall']:.2%}\n")
            f.write(f"- Overall F1 score: {overall_metrics['overall_f1']:.2%}\n")
            f.write(f"- Total matched items: {overall_metrics['total_matched']}\n")
            f.write(f"- Total reference items: {overall_metrics['total_reference']}\n")
            f.write(f"- Total test items: {overall_metrics['total_test']}\n")
            f.write(f"- Total missed items: {overall_metrics['total_missed']}\n")
            f.write(f"- Total extra items: {overall_metrics['total_extra']}\n\n")
            
            f.write("## Individual File Results\n\n")
            for test_file, data in results.items():
                metrics = data['metrics']
                
                f.write(f"### {test_file} (vs {data['reference_file']})\n\n")
                f.write(f"- Precision: {metrics['precision']:.2%}\n")
                f.write(f"- Recall: {metrics['recall']:.2%}\n")
                f.write(f"- F1 Score: {metrics['f1_score']:.2%}\n")
                f.write(f"- Matched items: {metrics['matched_count']} of {metrics['reference_count']}\n\n")
                
                # Missed items
                if data['missed_items']:
                    f.write("#### Missed Items (in reference but not detected)\n\n")
                    f.write("| Product | Reference Time |\n")
                    f.write("|---------|---------------|\n")
                    for item in data['missed_items']:
                        f.write(f"| {item['product']} | {item['ref_time']} |\n")
                    f.write("\n")
                
                # Extra items
                if data['extra_items']:
                    f.write("#### Extra Items (detected but not in reference)\n\n")
                    f.write("| Product | Test Time |\n")
                    f.write("|---------|----------|\n")
                    for item in data['extra_items']:
                        f.write(f"| {item['product']} | {item['test_time']} |\n")
                    f.write("\n")
                
                # Matched items
                if data['matched_items']:
                    f.write("#### Matched Items\n\n")
                    f.write("| Product | Test Time | Reference Time |\n")
                    f.write("|---------|-----------|---------------|\n")
                    for item in data['matched_items']:
                        f.write(f"| {item['product']} | {item['test_time']} | {item['ref_time']} |\n")
                    f.write("\n")
                
                f.write("\n")
                
            print(f"✅ Report saved to {output_file}")
    
    return {
        'overall_metrics': overall_metrics,
        'file_results': results
    }

def main():
    parser = argparse.ArgumentParser(description='Compare receipt CSV files')
    parser.add_argument('--test', required=True, help='Directory containing test receipt CSV files')
    parser.add_argument('--reference', required=True, help='Directory containing reference receipt CSV files')
    parser.add_argument('--output', help='Path to save evaluation report', default='receipt_evaluation_report.md')
    parser.add_argument('--tolerance', type=float, default=2.0, help='Time tolerance in seconds for matching')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.test):
        print(f"Error: Test directory '{args.test}' does not exist")
        return
        
    if not os.path.isdir(args.reference):
        print(f"Error: Reference directory '{args.reference}' does not exist")
        return
    
    evaluate_receipts(args.test, args.reference, args.output, args.tolerance)

if __name__ == "__main__":
    main()