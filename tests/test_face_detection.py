from face_detection import bbox_center


def test_bbox_center():
    cx, cy, (bx, by, bw, bh) = bbox_center(0.1, 0.2, 0.3, 0.4, 1000, 800)
    assert bx == 100
    assert by == 160
    assert bw == 300
    assert bh == 320
    assert cx == 100 + 150
    assert cy == 160 + 160
