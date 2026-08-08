from datasets import load_dataset

# Load the exact SFT dataset used for the 135M Instruct model.
# This contains subsets like smol-magpie-ultra-short, smol-constraints, smol-rewrite, etc.
dataset = load_dataset("HuggingFaceTB/smol-smoltalk")
for source in dataset.unique("source"):
    count = len(dataset.filter(lambda x: x["source"] == source))
    print(f"{source}: {count}")

# smol-magpie-ultra-short: 270838
# smol-contraints: 34433
# self-oss-instruct: 48071
# smollm-rewrite-30k: 26657
# openhermes-50k: 47492
# smol-summarize-20k: 19272
# longalign: 3560
# explore-instruct-rewrite: 3017
# smol-summarize-5k: 4749
# everyday-conversations: 2252