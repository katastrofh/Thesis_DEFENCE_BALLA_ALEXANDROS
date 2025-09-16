# microchain/runs/exp_reorg.py
# Note: the minimal validator does not implement reorg; this file shows the limitation and returns a note.
def run():
    return {"note":"The minimal demo keeps a single canonical head without reorg simulation."}

if __name__ == "__main__":
    print("Hi")