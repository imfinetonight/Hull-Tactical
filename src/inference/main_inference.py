from src.utils.setup import setup_environment
from src.inference.server import run_server


def main():
    setup_environment()
    run_server()


if __name__ == "__main__":
    main()