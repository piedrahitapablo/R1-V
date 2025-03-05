# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import subprocess

from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from transformers import TrainerCallback

from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainerModified
from trl import (
    GRPOConfig,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_peft_config,
)


class SyncS3CheckpointCallback(TrainerCallback):
    def __init__(self, s3_output_dir: Optional[str] = None):
        self.s3_output_dir = s3_output_dir

    def on_save(self, args, state, control, **kwargs):
        # `args`:       TrainingArguments (hyperparameters, output_dir, etc.)
        # `state`:      TrainerState (current epoch, global_step, etc.)
        # `control`:    TrainerControl (can be used to instruct the trainer to skip or interrupt actions)
        # `**kwargs`:   Additional properties like model, optimizer, lr_scheduler, etc.

        print(f">>Checkpoint is being saved. Current step: {state.global_step}")
        print("state", state)
        checkpoint_folder = f"checkpoint-{state.global_step}"

        if self.s3_output_dir is not None and args.output_dir is not None:
            print(os.system("ls -l"))
            print(">>>Syncing to S3 (not lora)")
            process = subprocess.Popen(
                f"aws s3 sync {os.path.join(args.output_dir, checkpoint_folder)} {os.path.join(self.s3_output_dir, checkpoint_folder)} && rm -rf {os.path.join(args.output_dir, checkpoint_folder)}",
                shell=True,
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Print output in real-time
            # while True:
            #     output = process.stdout.readline()
            #     error = process.stderr.readline()

            #     if output:
            #         print(f"Sync output: {output.decode().strip()}")
            #     if error:
            #         print(f"Sync error: {error.decode().strip()}")

            #     # Break if process has finished
            #     if process.poll() is not None:
            #         break

            # remove the `args.output_dir` directory content
            # os.system(f"rm -rf {args.output_dir}/*")


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format'"
        },
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    s3_output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "S3 output directory"},
    )


def accuracy_reward(completions, review_status_binary, **kwargs):
    print("-" * 100)
    print("Completions: ", completions)

    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, ground_truth in zip(contents, review_status_binary):
        reward = 0.0

        try:
            content_match = re.search(r"<answer>(.*?)</answer>", content)
            inferred_review_status_binary = (
                content_match.group(1).strip().lower()
                if content_match
                else content.strip()
            )

            if inferred_review_status_binary == ground_truth:
                reward = 1.0
        except Exception:
            pass

        rewards.append(reward)

        print(f"------------- {current_time} Accuracy reward: {reward} -------------")
        print(f"Content: {content}")
        print(f"Expected review status: {ground_truth}")

    return rewards


def format_reward(completions, review_status_binary, **kwargs):
    """Reward function that checks if the completion has a specific format and rewards based on word count."""
    pattern = r"<think>.*?<\/think>\s*<answer>(.*?)<\/answer>"

    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content in completion_contents:
        match = re.fullmatch(pattern, content, re.DOTALL | re.MULTILINE)
        if match:
            inferred_review_status_binary = (
                match.group(1).strip().lower() if match else content.strip()
            )
            rewards.append(
                1.0 if inferred_review_status_binary == review_status_binary else 0.0
            )
        else:
            rewards.append(0.0)

        print(
            f"------------- {current_time} Accuracy reward: {rewards[-1]} -------------"
        )
        print(f"Content: {content}")

    return rewards


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user sends an AI-generated "
    "image of a restaurant dish to be classified with its corresponding "
    "context (dish name, dish description, notes and other reference images) "
    'and the assistant classifies the image as "fail" or "pass". The assistant '
    "first thinks about the reasoning process in the mind and then provides "
    "the user with the answer. The image needs to be classified accoprding to "
    "the following criteria: \n"
    "1. Accuracy:\n"
    "  a. ingredient-missing\n"
    "  b. ingredient-to-be-removed\n"
    "  c. ingredient-type-incorrect\n"
    "  d. ingredient-color-incorrect\n"
    "  e. ingredient-shape-incorrect\n"
    "  f. ingredient-texture-incorrect\n"
    "  g. ingredient-size-incorrect\n"
    "  h. scale-incorrect\n"
    "  i. portion-size-incorrect\n"
    "  j. unit-quantity-incorrect\n"
    "  k. ingredient-amount-incorrect\n"
    "  l. reference-image-potentially-incorrect\n"
    # "  m. process-guidelines-not-followed\n"
    "  m. inaccurate-placement-of-ingredients\n"
    "2. Aesthetics:"
    "  a. light-direction-incorrect\n"
    "  b. shadow-shape-incorrect\n"
    "  c. shadows-too-dispersed\n"
    "  d. shadows-too-tight\n"
    "  e. shadows-too-dark\n"
    "  f. shadows-too-light\n"
    "  g. shadows-missing\n"
    "  h. highlights-missing\n"
    "  i. highlights-too-bright\n"
    "  j. highlights-too-dark\n"
    "  k. overexposed\n"
    "  l. underexposed\n"
    "  m. contrast-too-high\n"
    "  n. contrast-too-low\n"
    "  o. oversharpened\n"
    "  p. blurry\n"
    "  q. oversaturated\n"
    "  r. undersaturated\n"
    "  s. white-balance-incorrect\n"
    "  t. perspective-incorrect\n"
    "  u. container-incorrect\n"
    "  v. angle-incorrect\n"
    "  w. background-incorrect\n"
    "  x. ingredient-placed-unrealistically\n"
    "  y. masking-issues\n"
    "  z. editing-artifacts-visible\n"
    "  aa. food-container-boundary-issue\n"
    "The reasoning process and answer are enclosed "
    "within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

PROMPT_TEMPLATE = (
    "Dish name: {dish_name}\n"
    "Dish description: {dish_description}\n"
    "Dish notes: {dish_notes}\n"
    "Output format is first the thinking process in <think> </think> and final "
    'answer, only the label ("fail" or "pass"), in <answer> </answer> tags'
)


def make_conversation(example):
    return {
        "prompt": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": PROMPT_TEMPLATE.format(
                            dish_name=example["restaurant_item_name"],
                            dish_description=example["restaurant_item_description"],
                            dish_notes=example["restaurant_item_comments"],
                        ),
                    },
                ],
            },
        ],
        # this field is needed because the trainer expects a "image" column
        # "image": example["media"],
    }


def main(script_args, training_args, model_args):
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    dataset = dataset.map(make_conversation)

    trainer_cls = (
        Qwen2VLGRPOTrainer
        if not training_args.use_vllm
        else Qwen2VLGRPOVLLMTrainerModified
    )

    print("-" * 100)
    print("using: ", trainer_cls)
    print("model_args: ", model_args)
    print("training_args: ", training_args)
    print("-" * 100)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split]
        if training_args.eval_strategy != "no"
        else None,
        callbacks=[SyncS3CheckpointCallback(s3_output_dir=script_args.s3_output_dir)],
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
