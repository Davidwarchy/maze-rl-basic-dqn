I think that how rewards are engineered is important. 

I was getting interesting results when testing the dqn. I was wondering why there was a cap at 5.0 reward per episode. The reason for this is that we our reward structure is as follows: 
* 0.1 for any movement 
* 1.0 for reaching goal 
* maximum steps of 50 per episode, after which we reset. 
Again the RL algorithm will always be somehow fulfilling our specifications. It maximizes our result by marktiming. So I think that we might want to engineer our reward structure better. 
This could be by the number of steps it takes to get done. This can help. 
Or we could scale 


## On the reinforcement learning for the quadruped, 
is it possible to account for the moving average readings, so that we ignore the oscillation? 

Reinforcement Learning is good at catching loopholes in our thinking. It can help someone take a wholesome concept on an idea or project or something else. 

I think that we want to consider the number of steps taken and the reward given freely to the steps. You want to encourage forward movement with either or all of the following:
* A large goal reward larger than accumulated rewards from steps. Eg if each episode is 1000 steps and rewards for stepping is 0.1, the agent can marktime to get a reward of 100 = 1000*0.1 if the reward for goal is less than 100. So we want to have a reward significantly higher. Maybe a figure like 200. 
* A punishment for time, whose total magnitude is larger than the rewards accumalated by steps. For example considering 1000 steps per episdoe