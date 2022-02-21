from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
TEST_DATA_DIR = Path(TEST_DIR / "data").resolve()


def get_data(filename):
    return str(Path(TEST_DATA_DIR / filename))
