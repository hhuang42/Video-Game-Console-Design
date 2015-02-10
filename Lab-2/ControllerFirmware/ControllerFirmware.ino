
/*
  Pin bindings:
  00 - Square
  01 - Cross
  02 - Circle
  03 - Triangle
  04 - Switch
  05 - R1
  06 - R2
  07 - L2
  08 - L1
  09 - Up
  10 - Left
  11 - Right
  12 - Down
  13 - LED
  A0 - LYINV
  A1 - L3
  A2 - LX
  A3 - RY
  A4 - R3
  A5 - RXINV
 */
 
int SQUARE   = 0;
int CROSS    = 1;
int CIRCLE   = 2;
int TRIANGLE = 3;
int SWITCH   = 4;
int R1       = 5;
int R2       = 6;
int L2       = 7;
int L1       = 8;
int UP       = 9;
int LEFT     = 10;
int RIGHT    = 11;
int DOWN     = 12;
int LED      = 13;
int LYINV    = A0;
int L3       = A1;
int LX       = A2;
int RY       = A3;
int R3       = A4;
int RXINV    = A5;

typedef struct DigitalState {
  bool pressed;
  bool changed;
} DigitalState;

typedef struct Analog2dState {
  int x_value;
  int y_value;
} Analog2dState;

typedef struct Button {
  int port;
  DigitalState state;
} Button;

typedef struct Joystick{
  int x_port;
  bool x_inverted;
  int y_port;
  bool y_inverted;
  Analog2dState analog_state;
  DigitalState up_state;
  DigitalState right_state;
  DigitalState left_state;
  DigitalState down_state;
} Joystick;

typedef struct KeyBind{
  char key;
  bool is_mouse_key;
  DigitalState* trigger_state;
} KeyBind;

typedef struct MouseBind{
  int x_subtotal;
  int y_subtotal;
  Analog2dState* analog_state;
} MouseBind;

struct SwitchBind;

typedef struct BindSet{
  KeyBind* key_binds;
  int key_count;
  MouseBind* mouse_binds;
  int mouse_count;
  SwitchBind* switch_binds;
  int switch_count;
} BindSet;

typedef struct SwitchBind{
  DigitalState* switch_state;
  BindSet on_set;
  BindSet off_set;
} SwitchBind;


const int BUTTON_COUNT = 15;
const int JOYSTICK_COUNT = 2;

Button buttons[BUTTON_COUNT];
Button* b_square = &buttons[0];
Button* b_cross  = &buttons[1];
Button* b_circle  = &buttons[2];
Button* b_triangle= &buttons[3];
Button* r1= &buttons[4];
Button* r2= &buttons[5];
Button* r3= &buttons[6];
Button* l1= &buttons[7];
Button* l2= &buttons[8];
Button* l3 = &buttons[9];
Button* d_up = &buttons[10];
Button* d_left = &buttons[11];
Button* d_right = &buttons[12];
Button* d_down = &buttons[13];
Button* face_switch = &buttons[14];
Joystick joystick[JOYSTICK_COUNT];
Joystick* left_joystick = &joystick[0];
Joystick* right_joystick = &joystick[1];



void updateDigitalState(struct DigitalState& state, bool new_value){
  state.changed = (new_value != state.pressed);
  state.pressed = new_value;
}

void updateButton(struct Button& button){
  bool new_input = !digitalRead(button.port);
  updateDigitalState(button.state, new_input);
}

const long long DEAD_RADIUS_SQUARED = sq(64/16);

void updateJoystick(struct Joystick& joystick){
  Analog2dState* state = &(joystick.analog_state);
  state->x_value = 
    joystick.x_inverted ? 512 - analogRead(joystick.x_port)
                        : analogRead(joystick.x_port) - 512;
  state->y_value = 
    joystick.y_inverted ? 512 - analogRead(joystick.y_port)
                        : analogRead(joystick.y_port) - 512;

  bool outside_dead_zone = sq(state->x_value/16) + sq(state->y_value/16) 
      >= DEAD_RADIUS_SQUARED;

  updateDigitalState(joystick.up_state, outside_dead_zone &&
                     (state->y_value*2 >= abs(state->x_value)));
  updateDigitalState(joystick.right_state, outside_dead_zone &&
                     (state->x_value*2 >= abs(state->y_value)));
  updateDigitalState(joystick.down_state, outside_dead_zone &&
                     (state->y_value*-2 >= abs(state->x_value)));
  updateDigitalState(joystick.left_state, outside_dead_zone &&
                     (state->x_value*-2 >= abs(state->y_value)));
}

void portSetup(){
  b_square->port = SQUARE;
  b_cross->port = CROSS;
  b_circle->port = CIRCLE;
  b_triangle->port = TRIANGLE;
  r1->port = R1;
  r2->port = R2;
  r3->port = R3;
  l1->port = L1;
  l2->port = L2;
  l3->port = L3;
  d_up->port = UP;
  d_down->port = DOWN;
  d_left->port = LEFT;
  d_right->port = RIGHT;
  face_switch->port = SWITCH;
  left_joystick->x_port = LX;
  left_joystick->x_inverted = false;
  left_joystick->y_port = LYINV;
  left_joystick->y_inverted = true;
  right_joystick->x_port = RXINV;
  right_joystick->x_inverted = true;
  right_joystick->y_port = RY;
  right_joystick->y_inverted = false;
}

void setKeyboardKeyBind(struct KeyBind& bind, char key, 
                        struct DigitalState& trigger_state){
  bind.key = key;
  bind.is_mouse_key = false;
  bind.trigger_state = &trigger_state;
}

void setMouseKeyBind(struct KeyBind& bind, char key, 
                        struct DigitalState& trigger_state){
  bind.key = key;
  bind.is_mouse_key = true;
  bind.trigger_state = &trigger_state;
}

void setMouseBind(struct MouseBind& bind, 
                  struct Analog2dState& analog_state){
  bind.analog_state = &analog_state;
}

void setSwitchBind(struct SwitchBind& bind, 
                   struct DigitalState& switch_state){
  bind.switch_state = &switch_state;
}


BindSet v_binds;

const int v_key_count = 6;
KeyBind v_key_binds [v_key_count];

SwitchBind v_l2_switch;

const int v_l2_on_key_count = 12;
KeyBind v_l2_on_key_binds [v_l2_on_key_count];

SwitchBind v_l2_off_r2_switch;

const int v_l2_off_r2_on_key_count = 12;
KeyBind v_l2_off_r2_on_key_binds [v_l2_off_r2_on_key_count];

const int v_l2_off_r2_off_key_count = 9;
KeyBind v_l2_off_r2_off_key_binds [v_l2_off_r2_off_key_count];

const int v_l2_off_r2_off_mouse_count = 1;
MouseBind v_l2_off_r2_off_mouse_binds [v_l2_off_r2_off_mouse_count];



void bindSetSetup(){
  v_binds.key_count = v_key_count;
  v_binds.key_binds = v_key_binds;
  v_binds.mouse_count = 0;
  v_binds.switch_count = 1;
  v_binds.switch_binds = &v_l2_switch;

  BindSet* v_l2_on_binds = &v_l2_switch.on_set;
  v_l2_on_binds->key_count = v_l2_on_key_count;
  v_l2_on_binds->key_binds = v_l2_on_key_binds;
  v_l2_on_binds->mouse_count = 0;
  v_l2_on_binds->switch_count = 0;

  BindSet* v_l2_off_binds = &v_l2_switch.off_set;
  v_l2_off_binds->key_count = 0;
  v_l2_off_binds->mouse_count = 0;
  v_l2_off_binds->switch_count = 1;
  v_l2_off_binds->switch_binds = &v_l2_off_r2_switch;

  BindSet* v_l2_off_r2_on_binds = &v_l2_off_r2_switch.on_set;
  v_l2_off_r2_on_binds->key_count = v_l2_off_r2_on_key_count;
  v_l2_off_r2_on_binds->key_binds = v_l2_off_r2_on_key_binds;
  v_l2_off_r2_on_binds->mouse_count = 0;
  v_l2_off_r2_on_binds->switch_count = 0;

  BindSet* v_l2_off_r2_off_binds = &v_l2_off_r2_switch.off_set;
  v_l2_off_r2_off_binds->key_count = v_l2_off_r2_off_key_count;
  v_l2_off_r2_off_binds->key_binds = v_l2_off_r2_off_key_binds;
  v_l2_off_r2_off_binds->mouse_count = v_l2_off_r2_off_mouse_count;
  v_l2_off_r2_off_binds->mouse_binds = v_l2_off_r2_off_mouse_binds;
  v_l2_off_r2_off_binds->switch_count = 0;

  setKeyboardKeyBind(v_key_binds[0], KEY_UP_ARROW, 
                     left_joystick->up_state);
  setKeyboardKeyBind(v_key_binds[1], KEY_LEFT_ARROW, 
                     left_joystick->left_state);
  setKeyboardKeyBind(v_key_binds[2], KEY_RIGHT_ARROW, 
                     left_joystick->right_state);
  setKeyboardKeyBind(v_key_binds[3], KEY_DOWN_ARROW, 
                     left_joystick->down_state);
  setKeyboardKeyBind(v_key_binds[4], 'e', r1->state);
  setKeyboardKeyBind(v_key_binds[5], 'q', l1->state);

  setSwitchBind(v_l2_switch, l2->state);




  setKeyboardKeyBind(v_l2_on_key_binds[0],'4',b_square->state);
  setKeyboardKeyBind(v_l2_on_key_binds[1],'3',b_triangle->state);
  setKeyboardKeyBind(v_l2_on_key_binds[2],'2',b_circle->state);
  setKeyboardKeyBind(v_l2_on_key_binds[3],'1',b_cross->state);

  setKeyboardKeyBind(v_l2_on_key_binds[4], KEY_F1, 
                     right_joystick->down_state);
  setKeyboardKeyBind(v_l2_on_key_binds[5], KEY_F2, 
                     right_joystick->right_state);
  setKeyboardKeyBind(v_l2_on_key_binds[6], KEY_F3, 
                     right_joystick->up_state);
  setKeyboardKeyBind(v_l2_on_key_binds[7], KEY_F4,
                     right_joystick->left_state);

  setKeyboardKeyBind(v_l2_on_key_binds[8],'c',d_down->state);
  setKeyboardKeyBind(v_l2_on_key_binds[9],'v',d_left->state);
  setKeyboardKeyBind(v_l2_on_key_binds[10],'n',d_up->state);
  setKeyboardKeyBind(v_l2_on_key_binds[11],'m',d_right->state);

  setSwitchBind(v_l2_off_r2_switch, r2->state);

  setKeyboardKeyBind(v_l2_off_r2_on_key_binds[0],'8',b_square->state);
  setKeyboardKeyBind(v_l2_off_r2_on_key_binds[1],'7',b_triangle->state);
  setKeyboardKeyBind(v_l2_off_r2_on_key_binds[2],'6',b_circle->state);
  setKeyboardKeyBind(v_l2_off_r2_on_key_binds[3],'5',b_cross->state);

  setKeyboardKeyBind(v_l2_off_r2_on_key_binds[4],KEY_F9,d_down->state);
  setKeyboardKeyBind(v_l2_off_r2_on_key_binds[5],KEY_F12,d_left->state);
  setKeyboardKeyBind(v_l2_off_r2_on_key_binds[6],KEY_F11,d_up->state);
  setKeyboardKeyBind(v_l2_off_r2_on_key_binds[7],KEY_F10,d_right->state);


  setKeyboardKeyBind(v_l2_off_r2_on_key_binds[8], KEY_F5, 
                     right_joystick->down_state);
  setKeyboardKeyBind(v_l2_off_r2_on_key_binds[9], KEY_F6, 
                     right_joystick->right_state);
  setKeyboardKeyBind(v_l2_off_r2_on_key_binds[10], KEY_F7, 
                     right_joystick->up_state);
  setKeyboardKeyBind(v_l2_off_r2_on_key_binds[11], KEY_F8,
                     right_joystick->left_state);


  setKeyboardKeyBind(v_l2_off_r2_off_key_binds[0],'a',b_square->state);
  setKeyboardKeyBind(v_l2_off_r2_off_key_binds[1],'w',b_triangle->state);
  setKeyboardKeyBind(v_l2_off_r2_off_key_binds[2],'s',b_circle->state);
  setKeyboardKeyBind(v_l2_off_r2_off_key_binds[3],'d',b_cross->state);

  setKeyboardKeyBind(v_l2_off_r2_off_key_binds[4],'x',d_down->state);
  setKeyboardKeyBind(v_l2_off_r2_off_key_binds[5],'z',d_left->state);
  setKeyboardKeyBind(v_l2_off_r2_off_key_binds[6],'9',d_up->state);
  setKeyboardKeyBind(v_l2_off_r2_off_key_binds[7],'0',d_right->state);

  setMouseKeyBind(v_l2_off_r2_off_key_binds[8], MOUSE_LEFT, r3->state);
  setMouseBind(v_l2_off_r2_off_mouse_binds[0], 
               right_joystick->analog_state);


  

}

void setup(){
  portSetup();
  bindSetSetup();
  
  //start serial connection
  Keyboard.begin();
  Mouse.begin();
  
  //configure pin2 as an input and enable the internal pull-up resistor
  for(int i = 0; i < BUTTON_COUNT ; ++i){
     pinMode(buttons[i].port, INPUT_PULLUP);
  }
  
  pinMode(LED, OUTPUT); 
}

void releaseKeyBind(struct KeyBind& key_bind){
 if(key_bind.is_mouse_key){
    Mouse.release(key_bind.key);
  } else {
    Keyboard.release(key_bind.key); 
  }
}

void updateKeyBind(struct KeyBind& key_bind){
  if(key_bind.trigger_state->changed){
    if(key_bind.trigger_state->pressed){
      if(key_bind.is_mouse_key){
        Mouse.press(key_bind.key);
      } else {
        Keyboard.press(key_bind.key); 
      }
    } else {
      if(key_bind.is_mouse_key){
        Mouse.release(key_bind.key);
      } else {
        Keyboard.release(key_bind.key); 
      }
    }
  }
}



const int MOUSE_SCALING = 112;

void updateMouseBind(struct MouseBind& mouse_bind){
  mouse_bind.x_subtotal += mouse_bind.analog_state->x_value/4;
  mouse_bind.y_subtotal -= mouse_bind.analog_state->y_value/4;
  int x_delta = (mouse_bind.x_subtotal + 512) / MOUSE_SCALING - 
                512 / MOUSE_SCALING;
  int y_delta = (mouse_bind.y_subtotal + 512) / MOUSE_SCALING - 
                512 / MOUSE_SCALING;
  mouse_bind.x_subtotal = (mouse_bind.x_subtotal + 512) % MOUSE_SCALING - 
                          512 % MOUSE_SCALING;
  mouse_bind.y_subtotal = (mouse_bind.y_subtotal + 512) % MOUSE_SCALING - 
                          512 % MOUSE_SCALING;
  int accel = sq(x_delta) + sq(y_delta);
  Mouse.move(x_delta*accel, y_delta*accel, 0);
}

void releaseBindSet(struct BindSet& bind_set);

void releaseSwitchBind(struct SwitchBind& switch_bind){
  releaseBindSet(switch_bind.on_set);
  releaseBindSet(switch_bind.off_set);
};

void updateBindSet(struct BindSet& bind_set);

void updateSwitchBind(struct SwitchBind& switch_bind){
  if(switch_bind.switch_state->changed){
    if(switch_bind.switch_state->pressed){
      releaseBindSet(switch_bind.off_set);
    } else {
      releaseBindSet(switch_bind.on_set);
    }
  }
    if(switch_bind.switch_state->pressed){
      updateBindSet(switch_bind.on_set);
    } else {
      updateBindSet(switch_bind.off_set);
    }
}

void releaseBindSet(struct BindSet& bind_set){
  for(int i = 0; i < bind_set.key_count; ++i){
    releaseKeyBind(bind_set.key_binds[i]);
  }
  for(int i = 0; i < bind_set.switch_count; ++i){
    releaseSwitchBind(bind_set.switch_binds[i]);
  }
}

void updateBindSet(struct BindSet& bind_set){
  for(int i = 0; i < bind_set.key_count; ++i){
    updateKeyBind(bind_set.key_binds[i]);
  }
  for(int i = 0; i < bind_set.mouse_count; ++i){
    updateMouseBind(bind_set.mouse_binds[i]);
  }
  for(int i = 0; i < bind_set.switch_count; ++i){
    updateSwitchBind(bind_set.switch_binds[i]);
  }
}

int sensorVal = 1;
void loop(){
  //read the pushbutton value into a variable
  for(int i = 0; i < BUTTON_COUNT; ++i){
    updateButton(buttons[i]);
  }
    for(int i = 0; i < JOYSTICK_COUNT; ++i){
    updateJoystick(joystick[i]);
  }
  updateBindSet(v_binds);
  sensorVal = !sensorVal;
  // Keep in mind the pullup means the pushbutton's
  // logic is inverted. It goes HIGH when it's open,
  // and LOW when it's pressed. Turn on pin 13 when the 
  // button's pressed, and off when it's not:
  if (sensorVal == HIGH) {
    digitalWrite(13, LOW);
  } 
  else {
    digitalWrite(13, HIGH);
  }
}


