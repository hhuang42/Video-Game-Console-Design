#Lab Report 6

##Introduction

In this lab, a pong video game was implemented to use motion controls established from an earlier lab.

##Design

Pong was chosen as a game since it was relatively simple to implement. Furthermore, given the complex nature of the motion detection code, it seemed simplest to build the game on top of the existing motion detection code, so it was imperative that the game itself be not too difficult to code.

2 player pong was chosen since extension of motion detection to 2 players seemed more straightforward than developing realistic AI to play pong.

In order to expand the existing code to accomodate detecting two types of stimulus instead of just one, the code was modified to change the desired stimulus every cycle, such that two player's motions would be detected on alternating frames.

Markers were still used given the success and reliability from Lab 5. In addition to the blue marker, a green marker was used, once again because the green color was a continuous range in the HSV colorspace.

The game logic was chosen to run at the relatively poor framerate of the processing in order to allow players a chance to react given that their own actions had high latency.

Finally, the game display was placed directly on top of the motion detection output in order to demonstrate the connection between the visual location of the marker and the location of the paddle.

## Testing

The motion detection design was tested primarily through displaying both markers in front of the web cam and making sure that each respective paddle moved to the corresponding height of the marker tip.

The game itself could be tested relatively easily by just playing a few rounds.

## Results and Discussion

![alt-text](https://github.com/hhuang42/Video-Game-Console-Design/blob/master/Lab-6/game_screen1.png)

![alt-text](https://github.com/hhuang42/Video-Game-Console-Design/blob/master/Lab-6/game_screen2.png)

The program was able to implement pong using the motion controls.

The alternation between detecting green and blue marker worked well, and the horrible framerate/latency made any other drops in performance difficult to notice.

The slow game rate also seemed relatively fair given how great the latency was.

The game worked as expected as well, behaving as expected from a pong game.

## Conclusion

In this lab, motion controls were used to control the actions of two paddles in a game of Pong.

2 hours were spent on this lab.
