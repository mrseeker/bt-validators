import time
import random
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

bt.logging.set_trace(False)
bt.logging.debug = False
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
                DahoasRewardModel(path="/root/dahoas", device="cuda"),
                DiversityRewardModel(device="cuda"),
                PromptRewardModel(device="cuda"),
            ]

def run_step(prompt: str, name: str):
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
    for weight_i, reward_fn_i in zip(reward_weights, reward_functions):
        reward_i = reward_fn_i.apply(prompt, responses, name).to("cuda")
        rewards += weight_i * reward_i
        if bt.config.neuron.log_rewards:
            event[reward_fn_i.name] = reward_i.tolist()
        bt.logging.trace(str(reward_fn_i.name), reward_i.tolist())

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
    )

    base_text = augment_event['best']
    exclude = augment_event['uids']
    for k in range(4):

        # Get a followup question, given the summarized context.
        prompt = followup_prompt(base_text, i=k)
        followup_event = run_step(
            prompt=prompt,
            name='followup' + str(k),
        )
        exclude += followup_event['uids']

        # Ask the followup question, given the original context.
        prompt = answer_prompt(base_text, followup_event['best'])
        answer_event = run_step(
            prompt=prompt,
            name='answer' + str(k),
        )
        exclude += answer_event['uids']

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
