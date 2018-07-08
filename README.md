# retro-baselines-rudder

Applying the [RUDDER paper (arxiv.org/abs/1806.07857)](https://arxiv.org/abs/1806.07857) to Sonic.

The outcome of this experiment (after only a few hours of coding) is that it gets to a score of 2000 (out of 10000) and plateaus there. From the output videos the problem seems to be that the LSTM has problems learning to predict the reward function.

This code is [retro-baselines](https://github.com/openai/retro-baselines) modified to use the baselines_rudder code.

Some of the requirements needed are listed in the included requirements.txt file. Install with `pip install -r requirements.txt`.

![reward_plot](imgs/reward.png)
