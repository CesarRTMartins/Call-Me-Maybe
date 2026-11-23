import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .data_models import Args, FunctionDefinition
from .formatting import H, R, X


def load_json_list(path: str, key: str | None = None) -> list[Any]:
    records: list[Any] = json.loads(Path(path).read_text())
    if key is None:
        return records
    return [str(record[key]) for record in records]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="call-me-maybe",
        usage=("uv run -m src [-h] [-f FUNCTIONS_DEFINITION] [-i INPUT] "
               "[-o OUTPUT] [-m MODEL] [-t TIMEOUT] [-e]"),
        description="Generating JSON using small large language models",
        epilog="Made by ekramer",
    )
    parser.add_argument(
        "-f", "--functions_definition",
        help="JSON file with definitions for functions to be called",
        type=lambda path: [
            FunctionDefinition(**item)
            for item in load_json_list(path)
        ],
        required=True,
    )
    parser.add_argument(
        "-i", "--input",
        help="JSON file to read prompts from",
        type=lambda path: load_json_list(path, "prompt"),
        required=True,
    )
    parser.add_argument(
        "-o", "--output",
        help="file to dump generated JSON objects in",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "-m", "--model",
        help="LLM to download and use from Hugging Face",
        default="Qwen/Qwen3-0.6B",
    )
    parser.add_argument(
        "-t", "--timeout",
        help="time, in seconds, allowed for each prompt",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "-e", "--examine",
        help="print token information for prompt results",
        action="store_true",
    )
    return parser


def escape_prompt(prompt: str) -> str:
    return prompt.replace("\\", "\\\\").replace('"', '\\"')


def elapsed_label(started_at: float) -> str:
    elapsed = round(time.perf_counter() - started_at)
    minutes, seconds = divmod(elapsed, 60)
    return f"{H}Running time is {minutes} minutes and {seconds} seconds.{X}"


def main() -> int:
    parser = build_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        return 1

    try:
        arguments = parser.parse_args(namespace=Args())
    except OSError as error:
        message = f"{H + R}Error while loading program: {error}{X}"
        print(message, file=sys.stderr)
        return 1

    # Import after argument parsing so help remains lightweight.
    from .llm_interface import LLMInterface

    try:
        interface = LLMInterface(
            arguments.model,
            arguments.functions_definition,
        )
    except Exception as error:
        message = f"{H + R}Error while initializing LLM: {error}{X}"
        print(message, file=sys.stderr)
        return 1

    started_at = time.perf_counter()
    for prompt in arguments.input:
        try:
            generated = interface.process_prompt(
                escape_prompt(prompt),
                arguments.timeout,
                arguments.examine,
            )
            interface.dump(generated, arguments.output)
        except RuntimeError as error:
            print(
                f"{H + R}\nError while processing prompt: {error}{X}",
                file=sys.stderr,
            )
        except (json.JSONDecodeError, OSError, ValidationError) as error:
            print(
                f"{H + R}\nError while dumping output: {error}{X}",
                file=sys.stderr,
            )
        except KeyboardInterrupt:
            return 1

        print(elapsed_label(started_at))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
