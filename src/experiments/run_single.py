import sys

from src.main import main


def run(problem_identifier: str) -> None:
    main(["--problems", problem_identifier, *sys.argv[1:]])
