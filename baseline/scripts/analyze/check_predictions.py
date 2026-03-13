#!/usr/bin/env python3
"""
Kiểm tra kỹ 2 file predictions
"""

def main():
    file1 = "outputs/predictions/predictions_blind.txt"
    file2 = "outputs/submissions/submission_blind_nh.txt"
    
    # Đọc file 1
    pred1 = {}
    with open(file1, 'r') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            track_id = parts[0]
            plate_with_conf = parts[1]
            plate = plate_with_conf.split(';')[0]  # Bỏ confidence
            pred1[track_id] = plate
            if i <= 5:
                print(f"File1 line {i}: track={track_id}, plate={plate}")
    
    print(f"\nFile 1 tổng: {len(pred1)} tracks")
    
    # Đọc file 2
    print("\n" + "="*60)
    pred2 = {}
    with open(file2, 'r') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            track_id = parts[0]
            plate = parts[1]
            pred2[track_id] = plate
            if i <= 5:
                print(f"File2 line {i}: track={track_id}, plate={plate}")
    
    print(f"\nFile 2 tổng: {len(pred2)} tracks")
    
    # So sánh chi tiết 10 track đầu
    print("\n" + "="*60)
    print("SO SÁNH CHI TIẾT 10 TRACK ĐẦU TIÊN:")
    print("="*60)
    
    test_tracks = ['track_10002', 'track_10005', 'track_10010', 'track_10015', 
                   'track_10016', 'track_10020', 'track_10022', 'track_10024',
                   'track_10026', 'track_10031']
    
    same = 0
    diff = 0
    
    for track_id in test_tracks:
        p1 = pred1.get(track_id, 'N/A')
        p2 = pred2.get(track_id, 'N/A')
        match = "✓" if p1 == p2 else "✗"
        if p1 == p2:
            same += 1
        else:
            diff += 1
        print(f"{track_id}: {p1} vs {p2} {match}")
    
    print(f"\nTrong 10 track đầu: {same} giống, {diff} khác")
    
    # Đếm toàn bộ
    print("\n" + "="*60)
    print("THỐNG KÊ TOÀN BỘ:")
    print("="*60)
    
    total_same = 0
    total_diff = 0
    
    for track_id in pred1.keys():
        if track_id in pred2:
            if pred1[track_id] == pred2[track_id]:
                total_same += 1
            else:
                total_diff += 1
    
    total = total_same + total_diff
    print(f"Tổng tracks: {total}")
    print(f"Giống nhau: {total_same} ({total_same/total*100:.2f}%)")
    print(f"Khác nhau: {total_diff} ({total_diff/total*100:.2f}%)")

if __name__ == "__main__":
    main()
