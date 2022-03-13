# PolyWrap

This is just a small wrapper for running the [PolyCoder
model](https://github.com/VHellendoorn/Code-LMs)
([paper](https://arxiv.org/abs/2202.13169)) on some prompts with a
variety of temperatures etc. Setup:

1. Get the model and unpack it somewhere. You can find instructions in
   [VHellendoorn's Code-LMs](https://github.com/VHellendoorn/Code-LMs)
   repository.
2. Change `FASTDATA` in `polycoder.py` to match where you unpacked the
   model.
3. `pip install -r requirements.txt`

## Usage

```
$ python polycoder.py --help
usage: polycoder.py [-h] [-m MAX_TOKENS] [-n NUM_SAMPLES] [-p TOP_P]
                    [-t TEMPERATURE] [--num_gpus NUM_GPUS]
                    prompt_files [prompt_files ...]

positional arguments:
  prompt_files          Prompt files

optional arguments:
  -h, --help            show this help message and exit
  -m MAX_TOKENS, --max_tokens MAX_TOKENS
                        Max number of tokens to generate (default: 512)
  -n NUM_SAMPLES, --num_samples NUM_SAMPLES
                        Number of samples to generate per prompt (default: 10)
  -p TOP_P, --top_p TOP_P
                        Top p (default: 0.0)
  -t TEMPERATURE, --temperature TEMPERATURE
                        Temperature (default: 0.5)
  --num_gpus NUM_GPUS   Number of GPUs to use (default: autodetect)
```
