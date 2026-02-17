from __future__ import annotations

from benchmark import ale_env


def test_ale_env_sets_logger_mode_to_error(monkeypatch):
    calls = []

    class DummyLoggerMode:
        Error = "error"

    class DummyALEInterface:
        @staticmethod
        def setLoggerMode(mode):
            calls.append(mode)

        def __init__(self):
            self._lives = 3

        def setInt(self, _key, _value):
            return None

        def setFloat(self, _key, _value):
            return None

        def loadROM(self, _path):
            return None

        def getLegalActionSet(self):
            return [0, 1]

        def getMinimalActionSet(self):
            return [0]

        def lives(self):
            return int(self._lives)

    class DummyRoms:
        @staticmethod
        def get_rom_path(game):
            return f"/tmp/{game}.bin"

    monkeypatch.setattr(ale_env, "ALEInterface", DummyALEInterface)
    monkeypatch.setattr(ale_env, "LoggerMode", DummyLoggerMode)
    monkeypatch.setattr(ale_env, "roms", DummyRoms)

    _ = ale_env.ALEAtariEnv(ale_env.ALEEnvConfig(game="pong"))
    assert calls == ["error"]
