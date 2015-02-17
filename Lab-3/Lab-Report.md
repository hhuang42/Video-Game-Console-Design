#Lab Report 3

##Introduction

In this lab, the code of Cube2 was modified in order to collect information used for an informal analysis on the effectiveness of
two different controller configurations with regards to use in the video game. In particular, the effect of control schemes on
accuracy across different weapons was considered.

##Design and testing

The Cube2 code was modified to keep track of the shots fired by each weapon, as well as each hit performed by a weapon.

The controllers chosen were my own controller, controller 1, and Sebastian's controller, controller 2, because they provided similar interfaces, as they had face buttons on the right hand side for weapon changing, a left joystick for movement, a right joystick for camera, and triggers
for jumping and firing.
The main difference between the two controllers consisted of the right camera joystick being precise on controller 2, whereas
large movements on controller 1 led to movement that increased with the cube of the movement of the joystick.

For each controller,
there were 3 round of Capture the Flag on the Abbey map for each controller, where each team had 5 players, such that 4 players
on the ally side were all bots, as well as all 5 players on the opposing side. Every bot was set to difficulty 50, and the matches
were played out to 10 minutes every time.

Capture the Flag mode was chosen in order to effectively test how well the player can move around the map, rather than just
reacting to enemies moving around nearby, which may not actually require any movement if the player is adept enough at aiming.

In addition, the kill/death scores were also recorded.

## Results and Discussion

The weapon results were as follows:


|                       | chaingun | fist  | grenade launcher | pistol | rifle  | rocketlauncher | shotgun | Total  |
|-----------------------|----------|-------|------------------|--------|--------|----------------|---------|--------|
| Hit                   |          |       |                  |        |        |                |         |        |
| Controller 1 Run 1    | 2        | 0     | 0                | 15     | 0      | 0              | 7       | 24     |
| Controller 1 Run 2    | 15       | 0     | 1                | 7      | 1      | 2              | 17      | 43     |
| Controller 1 Run 3    | 1        | 2     | 2                | 3      | 2      | 3              | 15      | 28     |
| Controller 2 Run 1    | 0        | 0     | 0                | 6      | 6      | 1              | 0       | 13     |
| Controller 2 Run 2    | 20       | 0     | 0                | 1      | 1      | 3              | 10      | 35     |
| Controller 2 Run 3    | 2        | 0     | 0                | 1      | 3      | 3              | 1       | 10     |
| Total                 |          |       |                  |        |        |                |         |        |
| Controller 1 Run 1    | 27       | 0     | 1                | 152    | 2      | 6              | 16      | 204    |
| Controller 1 Run 2    | 104      | 46    | 4                | 54     | 9      | 12             | 36      | 265    |
| Controller 1 Run 3    | 40       | 128   | 7                | 35     | 11     | 10             | 22      | 253    |
| Controller 2 Run 1    | 0        | 1     | 3                | 30     | 14     | 5              | 1       | 54     |
| Controller 2 Run 2    | 196      | 0     | 4                | 19     | 13     | 25             | 17      | 274    |
| Controller 2 Run 3    | 29       | 0     | 4                | 9      | 23     | 11             | 6       | 82     |
|                       |          |       |                  |        |        |                |         |        |
| Controller 1 Hit      | 18       | 2     | 3                | 25     | 3      | 5              | 39      | 95     |
| Controller 1 Total    | 171      | 174   | 12               | 241    | 22     | 28             | 74      | 722    |
| Controller 1 Accuracy | 10.53%   | 1.15% | 25.00%           | 10.37% | 13.64% | 17.86%         | 52.70%  | 13.16% |
|                       |          |       |                  |        |        |                |         |        |
| Controller 2 Hit      | 22       | 0     | 0                | 8      | 10     | 7              | 11      | 58     |
| Controller 2 Total    | 225      | 1     | 11               | 58     | 50     | 41             | 24      | 410    |
| Controller 2 Accuracy | 9.78%    | 0.00% | 0.00%            | 13.79% | 20.00% | 17.07%         | 45.83%  | 14.15% |

It is notable that there exists weapons that both improved and worsened with a finer yet slower camera. In particular, close
ranged weapons like shotguns and fists did worse, whereas longer ranged weapons like rifles and pistols did better. Furthermore,
the total number of shots fired is also lower on the second controller.

This seems consistent with personal experience playing the game on the slower camera, where it was much more difficult to make
finer movements to move into position or pursue enemies to use close range weapons. Furthermore, a faster camera was much
more useful when enemies were strafing around in order to keep aim on them. However, the finer movemement did allow for
better aiming if the enemy was afar in general.

This observation also seems consistent with the kill death scores, where the first controller has a .56 KDA but has 2 flags per match, whereas the second
controller has a KDA of .68 but less than 1 flag per match. This suggests that it was difficult to move around effectively to
gather flags with a slower camera, and it lead to safer behavior sniping with long range weapons in the back, as opposed to actively
chasing the flag and using shotguns for short range damage.


## Conclusion

In this lab, informal tests were run across controller configurations that suggest that slower joystick speeds can increase
accuracy but can make short range weapons harder to use, as well as reducing mobility overall.

Data on bot accuracy was also recorded, but there was not sufficient time to perform analysis with those data values.

8 hours were spent on this lab.
