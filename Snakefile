
models = ["RNN", "CNN", "DeepFam"]
seeds = list(range(10))

ALL = expand("logs/experiments/runs/{model}/ckpts/seed{seed}.ckpt", model=models, seed=seeds)

rule all:
    input:
        ALL
        
rule train:
    output:
        "logs/experiments/runs/{model}/ckpts/seed{seed}.ckpt"
    shell:
        "python train.py "
        "model={wildcards.model} "
        "name={wildcards.model} "
        "trainer.gpus=[1] "
        "seed={wildcards.seed}"