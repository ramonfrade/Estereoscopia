from depth_config import DEFAULT_BASELINE_CM, DepthConfig, DepthMode


def test_depth_mode_values():
    assert DepthMode.FACE.value == "face"
    assert DepthMode.SGBM.value == "sgbm"


def test_depth_config_defaults():
    cfg = DepthConfig()
    assert cfg.baseline_cm == DEFAULT_BASELINE_CM
    assert cfg.focal_mm == 8.0
    assert cfg.fov_deg == 56.6


def test_depth_config_from_calibration_json(tmp_path):
    calib = tmp_path / "calibration.json"
    calib.write_text('{"physical_baseline_cm": 9.5}', encoding="utf-8")
    cfg = DepthConfig.from_calibration_json(calib)
    assert cfg.baseline_cm == 9.5


def test_depth_config_reads_calibration_error(tmp_path):
    calib = tmp_path / "calibration.json"
    calib.write_text(
        '{"physical_baseline_cm": 7.0, "calibration_error_pct": 1.23}',
        encoding="utf-8",
    )
    cfg = DepthConfig.from_calibration_json(calib)
    assert cfg.calibration_error_pct == 1.23


def test_depth_config_error_none_when_absent(tmp_path):
    calib = tmp_path / "calibration.json"
    calib.write_text("{}", encoding="utf-8")
    cfg = DepthConfig.from_calibration_json(calib)
    assert cfg.calibration_error_pct is None
