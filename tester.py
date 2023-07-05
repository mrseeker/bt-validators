import time
from random import random
from typing import List

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import bittensor as bt
from openvalidators.prompts import followup_prompt, answer_prompt, augment_prompt
import openvalidators
from openvalidators import dataset
from openvalidators.reward import OpenAssistantRewardModel, ReciprocateRewardModel, DahoasRewardModel, \
    DiversityRewardModel, PromptRewardModel
from openvalidators.reward.config import DefaultRewardFrameworkConfig

bt.logging.set_trace(True)
score = 0
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
reward_weights = torch.tensor([
                DefaultRewardFrameworkConfig.rlhf_model_weight, DefaultRewardFrameworkConfig.reciprocate_model_weight, DefaultRewardFrameworkConfig.dahoas_model_weight,
                DefaultRewardFrameworkConfig.diversity_model_weight, DefaultRewardFrameworkConfig.prompt_model_weight
            ], dtype=torch.float32).to("cuda")

reward_functions = [
                OpenAssistantRewardModel(device="cuda"),
                ReciprocateRewardModel(device="cuda"),
                DahoasRewardModel(path=openvalidators.config.neuron.full_path, device="cuda"),
                DiversityRewardModel(device="cuda"),
                PromptRewardModel(device="cuda"),
            ]

def run_step(self, prompt: str, k: int, timeout: float, name: str, exclude: list = []):
    global score
    bt.logging.debug("run_step", name)

    # Record event start time.
    event = {'name': name}
    start_time = time.time()

    input_ids = tokenizer(f"USER: {prompt}", return_tensors="pt").input_ids

    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.9,
        max_length=100,
    )
    responses = [tokenizer.batch_decode(gen_tokens)[0]]
    uids = [0]
    # Compute the rewards for the responses given the prompt.
    rewards: torch.FloatTensor = torch.zeros(len(responses), dtype=torch.float32).to("cuda")
    for weight_i, reward_fn_i in zip(openvalidators.reward_weights, openvalidators.reward_functions):
        reward_i = reward_fn_i.apply(prompt, responses, name).to("cuda")
        rewards += weight_i * reward_i
        if bt.config.neuron.log_rewards:
            event[reward_fn_i.name] = reward_i.tolist()
        bt.logging.trace(str(reward_fn_i.name), reward_i.tolist())

    for masking_fn_i in openvalidators.masking_functions:
        mask_i = masking_fn_i.apply(prompt, responses, name).to("cuda")
        rewards *= mask_i
        if openvalidators.config.neuron.log_rewards:
            event[masking_fn_i.name] = mask_i.tolist()
        bt.logging.trace(str(masking_fn_i.name), mask_i.tolist())

    # Find the best completion given the rewards vector.
    completions: List[str] = [comp.completion for comp in responses]
    best: str = completions[rewards.argmax(dim=0)].strip()

    # Get completion times
    completion_times: List[float] = [comp.elapsed_time for comp in responses]

    score = score + rewards.argmax(dim=0)
    # Log the step event.
    event.update({
        'uids': uids.tolist(),
        'step_length': time.time() - start_time,
        'prompt': prompt,
        'completions': completions,
        'completion_times': completion_times,
        'rewards': rewards.tolist(),
        'best': best
    })
    bt.logging.debug("event:", str(event))

    # Return the event.
    return event

def forward():
    global score
    # Obtain a unique context from the dataset.
    datas = dataset.Dataset()
    data = next(datas)["text"]

    random_cutoff = random.randint(15, 30)
    # Truncate context to a limited set of sentences.
    base_text = '.'.join(data.split('.', maxsplit=random_cutoff)[:-1])
    aug_prompt = augment_prompt(base_text)

    # Request a summary, given the original context.
    augment_event = run_step(
        prompt=aug_prompt,
        name='augment',
        k=openvalidators.neuron.followup_sample_size,
        timeout=openvalidators.config.neuron.followup_timeout,
    )

    base_text = augment_event['best']
    exclude = augment_event['uids']
    for k in range(openvalidators.neuron.num_followup_steps):

        # Get a followup question, given the summarized context.
        prompt = followup_prompt(base_text, i=k)
        followup_event = run_step(
            prompt=prompt,
            name='followup' + str(k),
            k=openvalidators.config.neuron.followup_sample_size,
            timeout=openvalidators.config.neuron.followup_timeout,
            exclude=exclude
        )
        exclude += followup_event['uids']

        # Ask the followup question, given the original context.
        prompt = answer_prompt(base_text, followup_event['best'])
        answer_event = run_step(
            prompt=prompt,
            name='answer' + str(k),
            k=openvalidators.config.neuron.answer_sample_size,
            timeout=openvalidators.config.neuron.answer_timeout,
            exclude=exclude
        )
        exclude += answer_event['uids']

        openvalidators.blacklist.question_blacklist.append(followup_event['best'])
        openvalidators.blacklist.answer_blacklist.append(answer_event['best'])

        if k == 0:
            # Extend the base text with the best answer.
            base_text = base_text + '\nPrevious Question \nQuestion:' + followup_event['best'] + '\nAnswer:' + \
                        answer_event['best']
        else:
            base_text = base_text + '\nQuestion:' + followup_event['best'] + '\nAnswer:' + answer_event['best']

for i in range(0, 10000):
    forward()
    if i % 100:
        print(f"Testing: {i}")
    print(f"SCORE: {score}")
