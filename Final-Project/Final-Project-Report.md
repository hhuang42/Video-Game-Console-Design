#Final Report

##Introduction

In this project, a basic bullet hell shooter was made that used GPU processing for game logic and CPU processing for game rendering. The game offers basic controls for a bullet hell shooter.

##Controls

| Key        | Use    |
|------------|--------|
| Z          | Shoots |
| Shift      | Focus  |
| Arrow Keys | Move   |

Focus reduces speed and causes firing to be more focused.

##Design

The game is a bullet hell shooter that uses CUDA in order to process bullet information in a bullet hell game shooter. All bullet processing was performed on the GPUs using CUDA kernels that were coded directly. During one game cycle, the GPU updated the location of the bullets, performed collision detection between the bullets and the player, and performed garbage collection on the memory of bullets that were no longer active in the game.

At the same time, the CPU ran the main script, manipulating enemies and generating new bullets, moving the players, rendering the game field using OpenGL, and updating the information in the terminal. Furthermore, new bullets were transferred to the GPU and bullet positions were transferred back to the CPU every cycle.

All data was stored in the form of fixed length arrays, and garbage collection was used to ensure that the arrays never overflowed. Several algorithms were used to optimize this process across multiple threads in the GPU. Rendering was performed on the CPU for simplicity.

Additional information and details can be found in the various presentations given about the project:

[Proposal](https://github.com/hhuang42/Video-Game-Console-Design/blob/master/Final-Project/MyProposal.pdf)

[Tech Talk](https://github.com/hhuang42/Video-Game-Console-Design/blob/master/Final-Project/TechTalk.pdf)

[Design Review](https://github.com/hhuang42/Video-Game-Console-Design/blob/master/Final-Project/Design%20Review.pdf)

[Tech Talk 2](https://github.com/hhuang42/Video-Game-Console-Design/blob/master/Final-Project/Tech%20Talk%202.pdf)

[Final Presentation](https://github.com/hhuang42/Video-Game-Console-Design/blob/master/Final-Project/FinalPresentation.pdf)

[Demo Video](https://www.dropbox.com/s/67lzh4xt1luwag5/demo.avi?dl=0)

The enemy patterns were inspired by the style of bullet patterns in games such as those in the _Touhou Project_ and other fangames made by myself.

## Future Work

In the future, it would be nice to implement fuller features such as bullet-destroying bombs and specialized bosses that would easily fit within the framework of the existing engine.
