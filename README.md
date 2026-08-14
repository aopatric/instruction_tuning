# instruction_tuning
Observing the impact of model maturity on the effectiveness of instruction tuning on Pythia at various training checkpoints.

Usage:

```bash
conda create -n instruct python=3.11
conda activate instruct
pip install -r requirements.txt
cd src
python training_demo.py # (args)
```