#Lab Report 1

##Introduction

In this lab, the hardware components of the controller were assembled. The controller was built with a wooden body, a breadboard mounted Arduino Micro, 8 face buttons, 2 analog sticks, 1 switch, and 4 shoulder buttons.

##Design methodology

The controller was designed primarily with action games in mind, while still allowing for FPS and fighting style games.

The left hand side of the controller places the joystick higher up on the controller for the purposes of controlling movement, as it personally feels less strenuous to keep the thumb higher on the controller for extended periods of time. Likewise, the right side of the controller places 4 face buttons higher up on the controller to allow easier access to commands in action games.

A joystick was added to the right hand side of the controller below the face buttons in order to allow precise camera control for FPS games, while face buttons were added to the left side of the controller to act like a d-pad, in the case that more precise directional inputs were required, such as in fighting games.

The shoulder buttons were placed primarily to act as camera control for action games, where the left thumb would be occupied by movement on the joystick and the right thumb would be occupied by commands on the face button.

However, a second pair of shoulder buttons were added to be analagous to L2 and R2 on PS2 controllers to act as modifiers on other button presses. Thus, holding down a modifier shoulder button while pressing a face button can perform a different action than a standalone button press. If the left hand is occupied by the joystick, the right hand can still press the 4 face buttons with either the left or right modifier active for an effective total of 8 additional commands that can be input. This can be very useful for playing PC-based action games that often use number keys as action shortcuts, such that there can be up to 10 shortcuts. Implementation of other combination actions like holding both modifier shoulder buttons can allow this controller to match the number of inputs required for those games without having 10 buttons dedicated for those shortcut slots.

The shoulder buttons were also implemented with smaller buttons than those used for the face buttons. This allowed two sets of buttons to fit on the shoulders. Furthermore, the small size is less noticeable because the trigger fingers constantly rest on the buttons, unlike the thumbs, which must traverse between various face buttons and joysticks and must locate the desired button by touch during swift movements.

Finally, a control switch was placed on the breadboard as the available switches had pins on the bottom of the switch, making it awkward to mount directly on the controller. Since the switch would most certainly be used to trigger modes and be expected to maintain its state during normal gameplay, the placement on the breadboard also makes it difficult to accidentally trigger the switch during normal gameplay.

Although the controller does not have dedicated Start/Select buttons, the joystick buttons were found to be sufficiently difficult to trigger and inconvenient to use that they can likely serve the role of Start/Select buttons.

