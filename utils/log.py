from enum import Enum


class LogLevelColor(Enum):
    CRITICAL = "\033[1;31m"
    DEBUG = "\033[0;36m"
    ERROR = "\033[0;31m"
    INFO = "\033[0;32m"
    WARNING = "\033[1;33m"


class Logger:
    @staticmethod
    def log(level: LogLevelColor, msg: str) -> None:
        separator = ":" + " " * (9 - len(level.name))
        prefix = f"{level.value}{level.name}{'\033[0m'}{separator}"
        print(f"{prefix}{msg}")

    @staticmethod
    def critical(msg: str) -> None:
        Logger.log(LogLevelColor.CRITICAL, msg)

    @staticmethod
    def error(msg: str) -> None:
        Logger.log(LogLevelColor.ERROR, msg)

    @staticmethod
    def warning(msg: str) -> None:
        Logger.log(LogLevelColor.WARNING, msg)

    @staticmethod
    def info(msg: str) -> None:
        Logger.log(LogLevelColor.INFO, msg)

    @staticmethod
    def debug(msg: str) -> None:
        Logger.log(LogLevelColor.DEBUG, msg)
