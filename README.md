# Running locally

## Prerequisites

Create conda environment and activate it:

```bash
conda env create -f environment.yml
conda activate assistant
conda install -c conda-forge sox
```

Download the allosaurus model:

```bash
python -m allosaurus.bin.download_model -m eng2102
```

## Running the server

```bash
python -m assistant_server.server
```

# Testing

## Generate sample bvh

```bash
python .\assistant_server\gesture_generation\generate.py -a .\data\samples\barefoot.wav -o .\data\zeggs\options.json -s .\data\zeggs\styles\old.bvh
```

Results should be in `data/results`

## Sample gpt

```bash
python .\assistant_server\api_clients\gpt.py
```

## Sample tts

```bash
python .\assistant_server\api_clients\speech.py
```


## Troubleshooting

If running on Vultr, check if cuda is available by running:

```python
import torch
torch.cuda.is_available()
```

If it returns `False`, run:

```bash
sudo bash /opt/nvidia/drivers/linux_nvidia_client.run --ui=none --no-questions
```