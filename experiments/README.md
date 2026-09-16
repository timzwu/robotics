# Experiments

Code and results for the robotics experiments in this repo. Each experiment folder contains its question,
method, results, and supporting data.

## Pick-and-place · Post #0

[Teaching a Robot to Pick Up a Block](https://timzwu.substack.com/p/teaching-a-robot-to-pick-up-a-block)
compares robot policies and language-model control on an SO-101 arm.

| Experiment | Question |
|---|---|
| [Model comparison](results/01_model_comparison/) | How do ACT, SmolVLA, π0.5, and Astra perform on the same task? |
| [Data scaling](results/02_data_scaling/) | How does the amount of demonstration data affect SmolVLA? |
| [Instruction following](results/03_instruction_following/) | Can the fine-tuned models follow instructions absent from their training data? |
| [Camera ablation](results/04_camera_ablation/) | How do overhead, wrist, and combined camera views compare? |
| [Execution method](results/05_execution_method/) | How does Real-Time Chunking compare with stop-and-go execution? |
| [Language-model control](results/06_general_llm/) | Can Astra control the arm through images and motion tools without demonstrations? |

[Setup, evaluation protocol, and commands](results/01_model_comparison/methods.md) ·
[Training log](results/training_runs.md) · [Trial data](results/all_trials_wk1.csv)

## Working here

- [`tools/`](tools/) holds the recording, training, evaluation, plotting, and video scripts.
- [`results/`](results/) holds experiment write-ups, CSVs, charts, and publication video copies.
- `checkpoints/` and `results/trials/` hold downloaded weights and raw footage; both are gitignored.

Start with the [environment and hardware setup](../notes/setup.md), then the method for the experiment you want
to run. New experiments will document their own setup and evaluation protocol as the work develops.
