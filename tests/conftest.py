import os
from pyannote.database import registry


def pytest_sessionstart(session):
    registry.load_database(os.path.join(os.path.dirname(__file__), "data/database.yml"))