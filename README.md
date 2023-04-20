# Running locally

## Prerequisites

Create conda environment and activate it:

```bash
conda env create -f environment.yml
conda activate assistant
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