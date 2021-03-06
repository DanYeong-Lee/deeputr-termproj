
experiments = ["cnntrfm-pe2-pool1-oddconv"]
seeds = list(range(10))

ALL = expand("logs/experiments/runs/{experiment}/ckpts/seed{seed}.ckpt", experiment=experiments, seed=seeds)

rule all:
    input:
        ALL
        
rule train:
    output:
        "logs/experiments/runs/{experiment}/ckpts/seed{seed}.ckpt"
    shell:
        "python train.py "
        "experiment={wildcards.experiment} "
        "trainer.gpus=[0] "
        "seed={wildcards.seed}"