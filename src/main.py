import sys
from cli import interactive_cli
from web import start_web

def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  uv run src/main.py cli <model.pvt>")
        print("  uv run src/main.py web <model.pvt>")
        return

    mode = sys.argv[1]
    model_path = sys.argv[2]

    if mode == "cli":
        interactive_cli(model_path)
    elif mode == "web":
        start_web(model_path)
    else:
        print("Unknown mode. Use 'cli' or 'web'.")

if __name__ == "__main__":
    main()