# KnightBot NNUE Training

KnightBot v0.9.0 uses a first-generation NNUE architecture:

- 768 binary piece-square inputs
- 128 hidden neurons
- clipped ReLU
- one centipawn output
- White-perspective score

The C++ engine can load the exported `.nnue` files directly.

## 1. Create the Python environment

From the KnightBot repository:

```powershell
cd tools\nnue

py -m venv .venv

.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

pip install -r requirements.txt
Verify PyTorch sees CUDA:
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
2. Extract positions from PGN games
Example:
python extract_positions.py games.pgn --output positions.fen --skip-opening-plies 8 --every 2
For an initial small test:
python extract_positions.py games.pgn --output positions-test.fen --max-positions 10000
3. Label positions with Stockfish
Example:
python label_stockfish.py `
    --stockfish "C:\Chess\stockfish.exe" `
    --input positions-test.fen `
    --output training-test.tsv `
    --depth 10 `
    --threads 1 `
    --hash 128
For long labeling runs use:
--resume
4. Validate the dataset
python check_dataset.py --data training-test.tsv
5. Run a tiny training test
python train.py `
    --data training-test.tsv `
    --output-dir runs\test `
    --epochs 2 `
    --batch-size 1024
The output folder will contain:
- checkpoint-last.pt
- checkpoint-best.pt
- knightbot-latest.nnue
- knightbot-best.nnue
- metrics.csv
6. Verify the exported network
python verify_nnue.py --network runs\test\knightbot-best.nnue
7. Load it in KnightBot
From the KnightBot CLI:
nnueload tools\nnue\runs\test\knightbot-best.nnue
nnue on
nnueinfo
startpos
evalcompare
The Python verifier and KnightBot evalcompare NNUE value should match exactly for the same FEN.
Recommended progression
Do not start with millions of positions.
Use:
1. 10,000 positions
2. 100,000 positions
3. 500,000 positions
4. 1,000,000+
5. several million once the whole pipeline is proven
For the first real network, around 500k–1M well-labeled positions is enough to determine whether the architecture is learning useful chess evaluation.
The final target can later be significantly larger.

---
