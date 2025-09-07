#include <Servo.h>

Servo myservo;  // create Servo object to control a servo
// twelve Servo objects can be created on most boards
int motorPos = 0;

void setup() {
  Serial.begin(115200);
  myservo.attach(9);  // attaches the servo on pin 9 to the Servo object

}

void loop() {
  int sensorValue = analogRead(A0);
  Serial.println(sensorValue);
  delay(50);
  motorPos = map(sensorValue, 0, 671, 0, 180);
  myservo.write(motorPos); 
}
