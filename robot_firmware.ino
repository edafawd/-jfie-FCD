// robot_firmware.ino — Elegoo V1 shield, word protocol @ 9600 for robot_brain.py
// Commands (newline-terminated): FORWARD, SLOW_FORWARD, BACKWARD, LEFT, RIGHT, STOP
// Reports IR sensors back as: "IR_L:<0/1>,IR_R:<0/1>"
// Safety: motors halt if no command arrives within TIMEOUT ms.

#define ENA 5    // Right motors PWM (speed)  -- VERIFY against your working sketch
#define ENB 6    // Left  motors PWM (speed)
#define IN1 7    // Right motors direction
#define IN2 8
#define IN3 9    // Left  motors direction
#define IN4 11
#define IR_L A0  // optional IR bump sensors; leave unwired -> reads 0
#define IR_R A1

const int FAST = 200, SLOW = 130, TURN = 180;
unsigned long lastCmd = 0, lastReport = 0;
const unsigned long TIMEOUT = 400;   // stop if Pi goes quiet (ms)

void setup() {
  pinMode(ENA,OUTPUT); pinMode(ENB,OUTPUT);
  pinMode(IN1,OUTPUT); pinMode(IN2,OUTPUT);
  pinMode(IN3,OUTPUT); pinMode(IN4,OUTPUT);
  pinMode(IR_L,INPUT); pinMode(IR_R,INPUT);
  Serial.begin(9600);
  halt();
}

void loop() {
  if (Serial.available()) {
    String c = Serial.readStringUntil('\n'); c.trim();
    exec(c); lastCmd = millis();
  }
  if (millis() - lastCmd > TIMEOUT) halt();          // safety
  if (millis() - lastReport > 200) {                 // report IR
    lastReport = millis();
    Serial.print("IR_L:"); Serial.print(digitalRead(IR_L));
    Serial.print(",IR_R:"); Serial.println(digitalRead(IR_R));
  }
}

void exec(String c) {
  if      (c == "FORWARD")      go(FAST,FAST, true,true);
  else if (c == "SLOW_FORWARD") go(SLOW,SLOW, true,true);
  else if (c == "BACKWARD")     go(FAST,FAST, false,false);
  else if (c == "LEFT")         go(TURN,TURN, false,true);   // pivot left
  else if (c == "RIGHT")        go(TURN,TURN, true,false);   // pivot right
  else                          halt();                      // STOP / unknown
}

// rightFwd/leftFwd = direction of each side
void go(int rPwm, int lPwm, bool rightFwd, bool leftFwd) {
  analogWrite(ENA, rPwm); analogWrite(ENB, lPwm);
  digitalWrite(IN1, rightFwd?HIGH:LOW); digitalWrite(IN2, rightFwd?LOW:HIGH);
  digitalWrite(IN3, leftFwd ?HIGH:LOW); digitalWrite(IN4, leftFwd ?LOW:HIGH);
}

void halt() {
  analogWrite(ENA,0); analogWrite(ENB,0);
  digitalWrite(IN1,LOW); digitalWrite(IN2,LOW);
  digitalWrite(IN3,LOW); digitalWrite(IN4,LOW);
}
