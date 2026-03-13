#!/usr/bin/env python3
"""
Phân tích chi tiết sự khác biệt giữa predictions_blind.txt và predictions_blind_enhanced.txt
"""

import sys
from collections import defaultdict

def load_predictions_with_confidence(file_path):
    """Load predictions with confidence scores"""
    predictions = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                track_id = parts[0]
                plate_conf = parts[1].split(';')
                plate = plate_conf[0]
                conf = float(plate_conf[1]) if len(plate_conf) > 1 else 0.0
                predictions[track_id] = {'plate': plate, 'confidence': conf}
    return predictions

def validate_brazilian_format(plate):
    """Validate Brazilian plate format"""
    if len(plate) != 7:
        return False, "invalid_length"
    
    # Brazilian old: XXX9999
    if plate[:3].isalpha() and plate[3:].isdigit():
        return True, "brazilian_old"
    
    # Mercosur: XXX9X99
    if (plate[:3].isalpha() and plate[3].isdigit() and 
        plate[4].isalpha() and plate[5:].isdigit()):
        return True, "mercosur"
    
    return False, "invalid_format"

def analyze_difference(plate1, plate2):
    """Analyze the type of difference between two plates"""
    if len(plate1) != len(plate2):
        return f"length_diff_{abs(len(plate1)-len(plate2))}"
    
    diff_positions = []
    diff_types = []
    
    for i, (c1, c2) in enumerate(zip(plate1, plate2)):
        if c1 != c2:
            diff_positions.append(i)
            
            # Check if it's a likely O/0, I/1, etc. confusion
            confusions = {
                ('O', '0'), ('0', 'O'), ('I', '1'), ('1', 'I'),
                ('B', '8'), ('8', 'B'), ('S', '5'), ('5', 'S'),
                ('G', '6'), ('6', 'G'), ('Z', '2'), ('2', 'Z'),
            }
            if (c1, c2) in confusions:
                diff_types.append('confusion')
            else:
                diff_types.append('different')
    
    if len(diff_positions) == 0:
        return "same"
    
    if all(t == 'confusion' for t in diff_types):
        return f"confusion_{len(diff_positions)}_chars"
    
    return f"different_{len(diff_positions)}_chars"

def main():
    file1 = "outputs/predictions/predictions_blind.txt"
    file2 = "outputs/predictions/predictions_blind_enhanced.txt"
    
    print("="*80)
    print("PHÂN TÍCH CHI TIẾT: predictions_blind vs predictions_blind_enhanced")
    print("="*80)
    
    pred1 = load_predictions_with_confidence(file1)
    pred2 = load_predictions_with_confidence(file2)
    
    print(f"\n📊 File 1 ({file1}): {len(pred1)} tracks")
    print(f"📊 File 2 ({file2}): {len(pred2)} tracks")
    
    # Statistics
    common_tracks = set(pred1.keys()) & set(pred2.keys())
    
    same_count = 0
    diff_count = 0
    differences = []
    
    # Format validation stats
    format_stats_1 = defaultdict(int)
    format_stats_2 = defaultdict(int)
    
    # Confidence comparison
    conf_same = []
    conf_diff = []
    
    # Difference analysis
    diff_analysis = defaultdict(int)
    
    for track_id in sorted(common_tracks):
        p1 = pred1[track_id]
        p2 = pred2[track_id]
        
        plate1 = p1['plate']
        plate2 = p2['plate']
        conf1 = p1['confidence']
        conf2 = p2['confidence']
        
        valid1, fmt1 = validate_brazilian_format(plate1)
        valid2, fmt2 = validate_brazilian_format(plate2)
        
        format_stats_1[fmt1] += 1
        format_stats_2[fmt2] += 1
        
        if plate1 == plate2:
            same_count += 1
            conf_same.append((conf1, conf2))
        else:
            diff_count += 1
            conf_diff.append((conf1, conf2, track_id, plate1, plate2))
            differences.append((track_id, plate1, plate2, conf1, conf2, valid1, valid2))
            
            # Analyze difference type
            diff_type = analyze_difference(plate1, plate2)
            diff_analysis[diff_type] += 1
    
    # Print results
    print(f"\n{'='*80}")
    print("TỔNG QUAN:")
    print(f"{'='*80}")
    print(f"Tracks chung:     {len(common_tracks)}")
    print(f"Giống nhau:       {same_count} ({same_count/len(common_tracks)*100:.2f}%)")
    print(f"Khác nhau:        {diff_count} ({diff_count/len(common_tracks)*100:.2f}%)")
    
    # Format validation
    print(f"\n{'='*80}")
    print("FORMAT VALIDATION:")
    print(f"{'='*80}")
    print(f"\nFile 1 ({file1}):")
    for fmt, count in sorted(format_stats_1.items(), key=lambda x: -x[1]):
        print(f"  {fmt:20s}: {count:4d} ({count/len(pred1)*100:.2f}%)")
    
    print(f"\nFile 2 ({file2}):")
    for fmt, count in sorted(format_stats_2.items(), key=lambda x: -x[1]):
        print(f"  {fmt:20s}: {count:4d} ({count/len(pred2)*100:.2f}%)")
    
    # Confidence comparison
    print(f"\n{'='*80}")
    print("CONFIDENCE COMPARISON:")
    print(f"{'='*80}")
    
    if conf_same:
        avg_conf1_same = sum(c[0] for c in conf_same) / len(conf_same)
        avg_conf2_same = sum(c[1] for c in conf_same) / len(conf_same)
        print(f"Same predictions ({len(conf_same)} tracks):")
        print(f"  File 1 avg confidence: {avg_conf1_same:.4f}")
        print(f"  File 2 avg confidence: {avg_conf2_same:.4f}")
    
    if conf_diff:
        avg_conf1_diff = sum(c[0] for c in conf_diff) / len(conf_diff)
        avg_conf2_diff = sum(c[1] for c in conf_diff) / len(conf_diff)
        print(f"\nDifferent predictions ({len(conf_diff)} tracks):")
        print(f"  File 1 avg confidence: {avg_conf1_diff:.4f}")
        print(f"  File 2 avg confidence: {avg_conf2_diff:.4f}")
    
    # Difference type analysis
    print(f"\n{'='*80}")
    print("PHÂN TÍCH LOẠI KHÁC BIỆT:")
    print(f"{'='*80}")
    for diff_type, count in sorted(diff_analysis.items(), key=lambda x: -x[1]):
        print(f"  {diff_type:30s}: {count:4d} ({count/diff_count*100:.2f}%)")
    
    # Show examples
    print(f"\n{'='*80}")
    print("EXAMPLES OF DIFFERENCES (30 đầu tiên):")
    print(f"{'='*80}")
    print(f"{'Track':<15} {'File 1':<12} {'File 2':<12} {'Conf1':>6} {'Conf2':>6} {'V1':>3} {'V2':>3} {'Type'}")
    print("-"*80)
    
    for track_id, plate1, plate2, conf1, conf2, valid1, valid2 in differences[:30]:
        v1 = '✓' if valid1 else '✗'
        v2 = '✓' if valid2 else '✗'
        diff_type = analyze_difference(plate1, plate2)
        print(f"{track_id:<15} {plate1:<12} {plate2:<12} {conf1:6.4f} {conf2:6.4f} {v1:>3} {v2:>3} {diff_type}")
    
    # Cases where enhanced has lower confidence but different prediction
    print(f"\n{'='*80}")
    print("CASES WHERE ENHANCED HAS LOWER CONFIDENCE:")
    print(f"{'='*80}")
    lower_conf_cases = [(tid, p1, p2, c1, c2) for c1, c2, tid, p1, p2 in conf_diff if c2 < c1]
    lower_conf_cases.sort(key=lambda x: x[3] - x[4], reverse=True)  # Sort by confidence drop
    
    print(f"Số lượng: {len(lower_conf_cases)} / {diff_count} ({len(lower_conf_cases)/diff_count*100:.2f}%)")
    print(f"\nTop 10 cases with largest confidence drop:")
    print(f"{'Track':<15} {'File 1':<12} {'File 2':<12} {'Conf1':>6} {'Conf2':>6} {'Drop':>6}")
    print("-"*80)
    for track_id, plate1, plate2, conf1, conf2 in lower_conf_cases[:10]:
        drop = conf1 - conf2
        print(f"{track_id:<15} {plate1:<12} {plate2:<12} {conf1:6.4f} {conf2:6.4f} {drop:6.4f}")
    
    # Cases where enhanced has higher confidence
    print(f"\n{'='*80}")
    print("CASES WHERE ENHANCED HAS HIGHER CONFIDENCE:")
    print(f"{'='*80}")
    higher_conf_cases = [(tid, p1, p2, c1, c2) for c1, c2, tid, p1, p2 in conf_diff if c2 > c1]
    higher_conf_cases.sort(key=lambda x: x[4] - x[3], reverse=True)  # Sort by confidence gain
    
    print(f"Số lượng: {len(higher_conf_cases)} / {diff_count} ({len(higher_conf_cases)/diff_count*100:.2f}%)")
    print(f"\nTop 10 cases with largest confidence gain:")
    print(f"{'Track':<15} {'File 1':<12} {'File 2':<12} {'Conf1':>6} {'Conf2':>6} {'Gain':>6}")
    print("-"*80)
    for track_id, plate1, plate2, conf1, conf2 in higher_conf_cases[:10]:
        gain = conf2 - conf1
        print(f"{track_id:<15} {plate1:<12} {plate2:<12} {conf1:6.4f} {conf2:6.4f} {gain:6.4f}")
    
    # Format improvements
    print(f"\n{'='*80}")
    print("FORMAT IMPROVEMENTS (Invalid → Valid):")
    print(f"{'='*80}")
    
    format_improvements = [d for d in differences if not d[5] and d[6]]  # invalid1 and valid2
    format_degradations = [d for d in differences if d[5] and not d[6]]  # valid1 and invalid2
    
    print(f"Improved (invalid→valid): {len(format_improvements)}")
    print(f"Degraded (valid→invalid): {len(format_degradations)}")
    
    if format_improvements:
        print(f"\nTop 10 format improvements:")
        print(f"{'Track':<15} {'File 1':<12} {'File 2':<12} {'Conf1':>6} {'Conf2':>6}")
        print("-"*80)
        for track_id, plate1, plate2, conf1, conf2, _, _ in format_improvements[:10]:
            print(f"{track_id:<15} {plate1:<12} {plate2:<12} {conf1:6.4f} {conf2:6.4f}")
    
    if format_degradations:
        print(f"\nTop 10 format degradations (⚠️  need investigation):")
        print(f"{'Track':<15} {'File 1':<12} {'File 2':<12} {'Conf1':>6} {'Conf2':>6}")
        print("-"*80)
        for track_id, plate1, plate2, conf1, conf2, _, _ in format_degradations[:10]:
            print(f"{track_id:<15} {plate1:<12} {plate2:<12} {conf1:6.4f} {conf2:6.4f}")
    
    print(f"\n{'='*80}")
    print("KẾT LUẬN:")
    print(f"{'='*80}")
    print(f"✅ Enhanced version có {len(format_improvements)} format improvements")
    print(f"⚠️  Enhanced version có {len(format_degradations)} format degradations")
    print(f"📊 Net format improvement: {len(format_improvements) - len(format_degradations)}")
    
    if len(higher_conf_cases) > len(lower_conf_cases):
        print(f"✅ Enhanced version có higher confidence trong {len(higher_conf_cases)} cases")
    else:
        print(f"⚠️  Enhanced version có lower confidence trong {len(lower_conf_cases)} cases")
    
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
