from golden_path_strands.logging import ExplorationLogger


def test_logger_tracks_success() -> None:
    logger = ExplorationLogger()
    logger.log_decision_point({"result": "foo"}, {"task": "t"})
    logger.mark_path_successful(0.5, {"meta": 1})
    paths = logger.get_successful_paths()
    assert paths and paths[0]["score"] == 0.5
