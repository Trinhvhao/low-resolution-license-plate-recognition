#!/usr/bin/env python3
"""
So sánh 2 file predictions để tìm chênh lệch
"""

def load_predictions(file_path):
    """Load predictions từ file"""
    predictions = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',')
            if len(parts) >= 2:
                track_id = parts[0]
                # Lấy plate (bỏ confidence score nếu có)
                plate = parts[1].split(';')[0]
                predictions[track_id] = plate
    
    return predictions

def compare_predictions(file1, file2):
    """So sánh 2 file predictions"""
    print(f"Đọc file 1: {file1}")
    pred1 = load_predictions(file1)
    print(f"  - Số lượng: {len(pred1)} tracks")
    
    print(f"\nĐọc file 2: {file2}")
    pred2 = load_predictions(file2)
    print(f"  - Số lượng: {len(pred2)} tracks")
    
    # Tìm các track_id chung
    common_tracks = set(pred1.keys()) & set(pred2.keys())
    only_in_file1 = set(pred1.keys()) - set(pred2.keys())
    only_in_file2 = set(pred2.keys()) - set(pred1.keys())
    
    print(f"\n{'='*60}")
    print(f"THỐNG KÊ:")
    print(f"{'='*60}")
    print(f"Tracks chung: {len(common_tracks)}")
    print(f"Chỉ có trong file 1: {len(only_in_file1)}")
    print(f"Chỉ có trong file 2: {len(only_in_file2)}")
    
    if only_in_file1:
        print(f"\nTracks chỉ có trong file 1 (10 đầu tiên):")
        for track_id in sorted(list(only_in_file1))[:10]:
            print(f"  {track_id}: {pred1[track_id]}")
    
    if only_in_file2:
        print(f"\nTracks chỉ có trong file 2 (10 đầu tiên):")
        for track_id in sorted(list(only_in_file2))[:10]:
            print(f"  {track_id}: {pred2[track_id]}")
    
    # So sánh các tracks chung
    same_count = 0
    diff_count = 0
    differences = []
    
    for track_id in sorted(common_tracks):
        plate1 = pred1[track_id]
        plate2 = pred2[track_id]
        
        if plate1 == plate2:
            same_count += 1
        else:
            diff_count += 1
            differences.append((track_id, plate1, plate2))
    
    print(f"\n{'='*60}")
    print(f"SO SÁNH TRÊN {len(common_tracks)} TRACKS CHUNG:")
    print(f"{'='*60}")
    print(f"Giống nhau: {same_count} ({same_count/len(common_tracks)*100:.2f}%)")
    print(f"Khác nhau: {diff_count} ({diff_count/len(common_tracks)*100:.2f}%)")
    
    if differences:
        print(f"\n{'='*60}")
        print(f"CHI TIẾT KHÁC BIỆT (20 đầu tiên):")
        print(f"{'='*60}")
        print(f"{'Track ID':<15} {'File 1':<12} {'File 2':<12} {'Note'}")
        print(f"{'-'*60}")
        
        for track_id, plate1, plate2 in differences[:20]:
            # Tính số ký tự khác nhau
            diff_chars = sum(c1 != c2 for c1, c2 in zip(plate1, plate2))
            len_diff = abs(len(plate1) - len(plate2))
            
            note = f"{diff_chars} chars"
            if len_diff > 0:
                note += f", len:{len_diff}"
            
            print(f"{track_id:<15} {plate1:<12} {plate2:<12} {note}")
        
        if len(differences) > 20:
            print(f"... và {len(differences) - 20} khác biệt khác")
    
    # Phân tích mức độ khác biệt
    if differences:
        print(f"\n{'='*60}")
        print(f"PHÂN TÍCH MỨC ĐỘ KHÁC BIỆT:")
        print(f"{'='*60}")
        
        char_diffs = []
        for track_id, plate1, plate2 in differences:
            diff_chars = sum(c1 != c2 for c1, c2 in zip(plate1, plate2))
            total_chars = max(len(plate1), len(plate2))
            char_diffs.append(diff_chars)
        
        avg_char_diff = sum(char_diffs) / len(char_diffs)
        print(f"Trung bình số ký tự khác/plate: {avg_char_diff:.2f}")
        
        # Đếm theo số ký tự khác
        diff_1_char = sum(1 for d in char_diffs if d == 1)
        diff_2_char = sum(1 for d in char_diffs if d == 2)
        diff_3_char = sum(1 for d in char_diffs if d == 3)
        diff_4plus = sum(1 for d in char_diffs if d >= 4)
        
        print(f"\nPhân bố khác biệt:")
        print(f"  1 ký tự: {diff_1_char} ({diff_1_char/len(differences)*100:.1f}%)")
        print(f"  2 ký tự: {diff_2_char} ({diff_2_char/len(differences)*100:.1f}%)")
        print(f"  3 ký tự: {diff_3_char} ({diff_3_char/len(differences)*100:.1f}%)")
        print(f"  4+ ký tự: {diff_4plus} ({diff_4plus/len(differences)*100:.1f}%)")
    
    return {
        'total_common': len(common_tracks),
        'same': same_count,
        'different': diff_count,
        'agreement_rate': same_count/len(common_tracks)*100 if common_tracks else 0,
        'difference_rate': diff_count/len(common_tracks)*100 if common_tracks else 0,
        'differences': differences
    }

if __name__ == "__main__":
    file1 = "outputs/predictions/predictions_blind.txt"
    file2 = "outputs/submissions/submission_blind_nh.txt"
    
    result = compare_predictions(file1, file2)
    
    print(f"\n{'='*60}")
    print(f"KẾT LUẬN:")
    print(f"{'='*60}")
    print(f"Tỷ lệ khớp: {result['agreement_rate']:.2f}%")
    print(f"Tỷ lệ khác biệt: {result['difference_rate']:.2f}%")
    print(f"{'='*60}")
