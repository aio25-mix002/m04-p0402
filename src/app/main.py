"""Entry point for a future web or command‑line interface.

Currently this file simply prints a message instructing the user to
execute the training script.  In a full application, you could use
frameworks such as Streamlit or FastAPI here to build interactive
dashboards for exploring the model outputs.
"""

def main() -> None:
    print(
        "This project is organised for reproducible machine learning. "
        "To train the model, run `python scripts/train.py` from the project root."
    )


if __name__ == "__main__":
    main()
