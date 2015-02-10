#Lab Report 1

##Introduction

In this lab, the firmware for the game controller was programmed to act as a set of USB mouse and keyboard in order to provide input to games on the computer.

##Design

The firmware was designed, in conjuction with the hardware, to provide enough controls to adequately play an online action RPG, Vindictus, in conjuction with providing controls that would be consistent with an FPS game. Since games often allow keys to be rebinded as desired, the actual key combination send over USB was not crucial as long as there existed some binding associated with the desired physical buttons. As a result, the key inputs were chosen to correspond with the existing key bindings for Vindictus in order to avoid reconfiguring those key inputs.

As a result, the key inputs were chosen as below:

| Key Input                 |  USB Output  | Vindictus Control | FPS Control       |
|---------------------------|:------------:|-------------------|-------------------|
| Square                    |       a      | Block/Dodge       | Cycle Left        |
| Triangle                  |       w      | Special Command   | Use Item          |
| Circle                    |       s      | Normal Attack     | Cycle Right       |
| Cross                     |       d      | Smash Attack      | Jump              |
| R1                        |       e      | Move Camera Right | Fire              |
| L1                        |       q      | Move Camera Left  | Walk/Run          |
| Left Joystick/Up          |   Up Arrow   | Movement Forward  | Movement Forward  |
| Left Joystick/Left        |  Left Arrow  | Movement Left     | Movement Left     |
| Left Joystick/Down        |  Down Arrow  | Movement Backward | Movement Backward |
| Left Joystick/Right       |  Right Arrow | Movement Right    | Movement Right    |
| Right Joystick            | Mouse Cursor | Mouse Cursor      | Camera            |
| R3                        |  Mouse Click | Mouse Click       | N/A               |
| D-Pad/Up                  |       9      | Shortcut Key      | N/A               |
| D-Pad/Right               |       0      | Shortcut Key      | N/A               |
| D-Pad/Left                |       z      | Transform         | N/A               |
| D-Pad/Right               |       x      | Toggle Form       | N/A               |
| L2 + Cross                |       1      | Shortcut Key      | N/A               |
| L2 + Circle               |       2      | Shortcut Key      | N/A               |
| L2 + Triangle             |       3      | Shortcut Key      | N/A               |
| L2 + Square               |       4      | Shortcut Key      | N/A               |
| R2 + Cross                |       5      | Shortcut Key      | N/A               |
| R2 + Circle               |       6      | Shortcut Key      | N/A               |
| R2 + Triangle             |       7      | Shortcut Key      | N/A               |
| R2 + Square               |       8      | Shortcut Key      | N/A               |
| L2 + D-Pad/Up             |       n      | Story Tab         | N/A               |
| L2 + D-Pad/Right          |       m      | Mission Tab       | N/A               |
| L2 + D-Pad/Left           |       v      | Skill Tab         | N/A               |
| L2 + D-Pad/Right          |       c      | Character Tab     | N/A               |
| R2 + D-Pad/Up             |      F11     | Sit               | N/A               |
| R2 + D-Pad/Right          |      F10     | Emote             | N/A               |
| R2 + D-Pad/Left           |      F12     | Continue          | N/A               |
| R2 + D-Pad/Right          |      F9      | Emote             | N/A               |
| L2 + Right Joystick/Up    |      F3      | Message           | N/A               |
| L2 + Right Joystick/Left  |      F4      | Message           | N/A               |
| L2 + Right Joystick/Down  |      F1      | Message           | N/A               |
| L2 + Right Joystick/Right |      F2      | Message           | N/A               |
| R2 + Right Joystick/Up    |      F7      | Emote             | N/A               |
| R2 + Right Joystick/Left  |      F6      | Emote             | N/A               |
| R2 + Right Joystick/Down  |      F8      | Emote             | N/A               |
| R2 + Right Joystick/Right |      F5      | Emote             | N/A               |

The codeflow within the firmware was constructed to first poll for the digital and analog states of the inputs and storing
the current state of the inputs, followed by checking each keybind to see if the necessary key combinations to activate the
keybind was present.

Digital inputs were stored as a single boolean state, and analog inputs were stored as a pair of ints. The digital inputs were read fairly straightforwardly, and the analog inputs could be read both as analog controls primarily for the mouse as well as a simulated d-pad that could support up to 4 buttons where 2 neighboring buttons could simultaneously be pressed by holding the joystick diagonally.

Mouse velocity was scaled to the cubic power of the analog values in order to allow both minute movement while still being able to mvoe the mouse across the screen within a second.

## Testing

The firmware was tested primarily through playthrough within Vindictus, particularly since the FPS playthrough testing will take place next week. All controls were used and tested if they could be used on command, and a mission was played through with the controller.

## Results and Discussion

The firmware was programmed and uploaded onto the controller successfully. The controller was able to be used to play Vindictus on the computer fairly reasonably, but it had a relatively high learning curve since I was accustomed to playing on the keyboard before, and I had no muscle memory for the controller use.

However, whenever the button combinations were recalled correctly, the controller responded well and the input went through as expected.

## Conclusion

In this lab, the firmware for the USB controller was designed, implemented, and uploaded.

8 hours were spent on this lab.