# Notes on the Maze
## State 
5x5 maze represented by an array of shape (5,5), ie, 5 rows by 5 columns 
## Rewards 
Assume a simple reward system: 
* -1.0 for hitting wall
* +0.1 for movement 
* +1.0 for reaching goal 
## Epsilon 
There's a problem when we reduce epsilon while not considering future rewards. What exactly is the problem? If we don't reach the goal, the system will incentivize going around in loops. 