import os
import json
import subprocess
from transformers import GPT2TokenizerFast
from string import Template
import fcntl

os.environ['DOCKER_HOST'] = 'unix:///var/run/docker-nvme.sock'

NUM_GPUS = 2

FASTDATA = '/fastdata'
BASEDIR = os.path.join(FASTDATA, 'CodeLMs')
INDIR  = os.path.join(BASEDIR, 'conf')
OUTDIR = os.path.join(BASEDIR, 'out')
DOCKER_IMG = 'polycoder:local'
VOCAB_FILE = os.path.join(BASEDIR, 'data/code-vocab.json')
MERGE_FILE = os.path.join(BASEDIR, 'data/code-merges.txt')

tok = GPT2TokenizerFast(VOCAB_FILE, MERGE_FILE)

MAX_TOKENS = 2048

def trim_prompt(prompt, n):
    """
    Trim a prompt to fit within the PolyCoder prompt length limit.
    Trims one line at a time.
    prompt: a string
    n: the number of tokens we want to generate

    Returns: a trimmed prompt such that len(tokenize(prompt)) + n <= MAX_TOKENS
    """
    tokens = tok.encode(prompt)
    if len(tokens) + n <= MAX_TOKENS: return prompt
    tokens = tokens[-(MAX_TOKENS-n+1):]
    token_strs = [tok.decode([t]) for t in tokens]
    try:
        first_nl = next(i for i in range(len(token_strs)) if '\n' in token_strs[i])
    except StopIteration:
        # No newlines in prompt, and the prompt is too big
        raise ValueError(f"Prompt cannot be trimmed to fit within {MAX_TOKENS} tokens")
    
    # Potential concern: if the last token containing a newline had trailing characters
    # after the newline, we might accidentally lop off those characters. But I checked
    # that for the PolyCoder tokenizer all the tokens with newlines have no trailing
    # characters.
    trimmed_prompt = ''.join(token_strs[first_nl+1:])
    return trimmed_prompt

# How to invoke Docker
DOCKER_CMD = [
    'nvidia-docker', 'run',
    '--privileged',
    '--rm', '-it',
    # '-e', 'CUDA_VISIBLE_DEVICES=${gpu_num}',
    '--shm-size=1g',
    '--ulimit', 'memlock=-1',
    '--mount', f'type=bind,src={BASEDIR}/checkpoints-2-7B,dst=/gpt-neox/checkpoints',
    # NB: want host and container paths to be the same to avoid
    # having to translate between them
    '-v', f'/fastdata:/fastdata',
    DOCKER_IMG,
]

CONTAINER_CMD = [
    './deepy.py',
    'generate.py',
    '${config}',
    'checkpoints/configs/local_setup.yml',
    'checkpoints/configs/2-7B.yml'
]

def template_cmd(cmd, **kwargs):
    return [Template(c).safe_substitute(**kwargs) for c in cmd]

# Atomic counter for naming files
def get_counter():
    cfname = os.path.join(BASEDIR, 'counter.txt')
    try:
        counter_file = open(cfname, 'r+')
        fcntl.flock(counter_file, fcntl.LOCK_EX)
        try:
            counter = int(counter_file.read())
        except ValueError as ve:
            print(ve)
            counter = 0 
    except FileNotFoundError:
        counter_file = open(cfname, 'w+')
        counter = 0
 
    counter_file.seek(0)
    counter_file.write(str(counter+1))
    fcntl.flock(counter_file, fcntl.LOCK_UN)
    counter_file.close()
    return counter

def prepare_cmd(prompt, max_tokens, temperature, top_p, n, gpu_num=0):
    """
    Prepares the codegen command to be run in a Docker container.

    Returns (cmd, outfile)
    """
    counter = get_counter()
    top_k = 0
    # Save the prompt to a file

    filename_pattern = f'PolyCoder_t{temperature:.2f}_p{top_p:.2f}_n{n}_max{max_tokens}.{counter:03d}'
    prompt_file = os.path.join(INDIR, f'Prompt_{filename_pattern}.txt')
    with open(prompt_file, 'w') as f:
        f.write(prompt)
    output_file = os.path.join(OUTDIR, f'Gen_{filename_pattern}.jsonl')
     
    TEXTGEN_CONFIG = {
        # Text gen type: `input-file`, `unconditional` or `interactive`
        "text-gen-type": "input-file",
        
        # Params for all
        "maximum_tokens": max_tokens,
        "temperature": temperature,  # Raise for higher sample-counts.
        "top_p": top_p,
        "top_k": top_k,
        "recompute": False,
        
        # `unconditional`/`input-file`: samples
        "num-samples": n,

        # input/output file
        "sample-input-file": prompt_file,
        "sample-output-file": output_file,

        # DeepSpeed doesn't respect CUDA_VISIBLE_DEVICES, so we need to set this
        "include": f"localhost:{gpu_num}",

        # Magic: even though we're doing inference, DeepSpeed still checks
        # that train_batch_size == micro_batch_per_gpu * gradient_acc_step * world_size64
        "train_batch_size": 32,
    }

    # Save the config to a file. Extension is YML but it's really JSON
    config_file = os.path.join(INDIR,
        f'Config_{filename_pattern}.yml')
    with open(config_file, 'w') as f:
        json.dump(TEXTGEN_CONFIG, f)

    # Run the container
    cmd = template_cmd(DOCKER_CMD + CONTAINER_CMD,
        config=config_file, gpu_num=gpu_num)
    print(cmd)
    return cmd, output_file

def create(prompt, max_tokens, temperature, top_p, n):
    cmd, output_file = prepare_cmd(prompt, max_tokens, temperature, top_p, n)
    subprocess.run(cmd)

    # Read the output
    results = []
    with open(output_file, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    return results

def create_batch(batch):
    output_files = []
    for i in range(0, len(batch), NUM_GPUS):
        procs = []
        for j, (prompt, max_tokens, temperature, top_p, n) in enumerate(batch[i:i+NUM_GPUS]):
            cmd, output_file = prepare_cmd(prompt, max_tokens, temperature, top_p, n, gpu_num=j)
            output_files.append(output_file)
            # procs.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
            procs.append(subprocess.Popen(cmd))
        for proc in procs:
            proc.wait()
    return output_files

if __name__ == "__main__":
    two_commands = [
        (TEST_PROMPT,   256, 0.5, 0.0, 10),
        (TEST_PROMPT_2, 256, 0.5, 0.0, 10),
    ]
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--max_tokens', type=int, default=512, help='Max number of tokens to generate')
    parser.add_argument('-n', '--num_samples', type=int, default=10, help='Number of samples to generate')
    parser.add_argument('-p', '--top_p', type=float, default=0.0, help='Top p')
    parser.add_argument('-t', '--temperature', type=float, default=0.5, help='Temperature')
    parser.add_argument('prompt_files', nargs='+', help='Prompt files')

    # j = create(TEST_PROMPT, max_tokens=100, temperature=0.5, top_p=0.0, n=10)
    result_files = create_batch(two_commands)
    print("\n".join(result_files))
