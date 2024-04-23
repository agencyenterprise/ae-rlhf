# Reinforcement Learning from Human Feedback (RLHF)

Reinforcement Learning from Human Feedback (RLHF) is the state-of-the-art method for 
aligning Large Language Models (LLM) as of early 2024.  But RLHF is a complicated  and 
involves training 2 models - the reward model and the LLM.  The reward model
itself is ussually an LLM with the unembbed head replaced with a reward predictor.
Therefore it can be costly and time-consuming to do any type of experimenting with 
RLHF for LLMs.  

This repository implements RLHF on gymnasium compatible environments. We designed 
the implementation with two goals
1. It should be easy for a single person to run a full RLHF process end-to-end 
2. It should be easy for students and researchers to understand the code.

To this end we implemented a *synchronous* training environment which has been 
boostrapped from [CleanRL](https://github.com/vwxyzjn/cleanrl/tree/master).  Users can
run the Cartpole or Atari examples out of the box, or use those folders to bootstrap
new experiments on other environments.

## Instructions
First clone the repository and ensure all the dependencies are installed.  If using
poetry you can run `poetry install .` from inside the repository directory.

Next, start the labeling UI and leave that running througout.
```python asgi.py```


### CartPole Example
To run the CartPole example complete the following steps in a new terminal
1. `cd examples/cartpole`
2. Run the data collection for the pretraining data. Provide a globally unique name
for the run.
    * `python collect_pretrain_data.py --name cartpole-01`
3. Open `localhost:8000` in your browser and click `cartpole-01` to start labeling.
4. Label all the pairs (see labeling below).
5. Pretrain the reward model using the data we just labeled
    * `python pretrain_reward_model.py --name cartpole-01`
6. When the pretraining finishes, run the final script to iteratively train the agent.
    * `python train_ppo_agent.py --name cartpole-01`

Follow the instructions, after each training iteration you should refresh the labeling
UI in the browser to get a fresh set of data to label.  Continue to label pairs until
training completes or until you are happy with the agent's performance.

### Atari Example
The atari example is similar to the carpole example, except you should pass in the 
environment id associated to the Atari game. The following example shows how to do this
for the Donkey Kong game.
1. `cd examples/atari`
2. Run the data collection for the pretraining data. Provide a globally unique name
for the run.
    * `python collect_pretrain_data.py --name donkeykong-01 --env-id ALE/DonkeyKong-v5`
3. Open `localhost:8000` in your browser and click `donkeykong-01` to start labeling.
4. Label all the pairs (see labeling below).
5. Pretrain the reward model using the data we just labeled
    * `python pretrain_reward_model.py --name donkeykong-01`
6. When the pretraining finishes, run the final script to iteratively train the agent.
    * `python train_ppo_agent.py --name donkeykong-01 --env-id ALE/DonkeyKong-v5`



#### Labeling
To label the data look at the samples collected determine which of the two clips 
is better.  If agent doesn't take any clear actions that are "better" in either clip
click `tie`.  If you really can't tell whats going on click "unknown" but be aware that
these samples will be dropped from training entirely.



#### Dev Mode
Once you collect some data it will be stored in the database.  This repository does 
not currently provide functionality to easily or directly manipulate data in this 
database - that includes deleting runs.  If you want to just test things out you can 
append `ENV=dev` to all the python scripts (including `ENV=dev python asgi.py`) which
will use the "dev database".  This dev database is great for testing ideas and can be
wiped by just removing the directory (by default ~/.ae_rlhf/dev).  Below is a full
example of running cartpole in dev mode.


First start the server
`ENV=dev python asgi.py`

In a new terminal.
1. `cd examples/cartpole`
2. Run the data collection for the pretraining data. Provide a globally unique name
for the run.
    * `ENV=dev python collect_pretrain_data.py --name cartpole-01 --n-pairs 5`
3. Open `localhost:8000` in your browser and click `cartpole-01` to start labeling.
4. Label all the pairs (see labeling below).
5. Pretrain the reward model using the data we just labeled
    * `ENV=dev python pretrain_reward_model.py --name cartpole-01`
6. When the pretraining finishes, run the final script to iteratively train the agent.
    * `ENV=dev python train_ppo_agent.py --name cartpole-01`

## App Structure
If you'd like to modify this code in anyway, it might be helpful to understand the 
basic structure of the repository.  In particular, the `app` directory is responsible
for running the labeling frontend and backend, which includes a FastAPI server, a 
sqlite database, and local file storage (for videos and observations for training
the reward model).  The app is intended to be run on a single machine and by default
the data is saved to `~/.ae_rlhf/<ENV>` where <ENV> is one of `dev` or `prod`.  The 
default is `prod` but this can be set to `dev` by prepending `ENV=dev` to any command 
(see dev mode above).

If you'd like to modify this to use cloud storage or other database backends this is 
possible but you will need to modify or extend the `load_*` and `save_*` functions in
`ae_rlhf.utils` to accomodate this, as well as updating the default save location set in 
`ae_rlhf.config` (or with an env variable)




## References
1. [Deep reinforcement learning from human preferences](https://arxiv.org/abs/1706.03741)
2. [CleanRL Paper](https://www.jmlr.org/papers/v23/21-1342.html)
3. [CleanRL Repo](https://github.com/vwxyzjn/cleanrl?tab=readme-ov-file)
