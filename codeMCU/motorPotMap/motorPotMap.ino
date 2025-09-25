#include <ESP32Servo.h>

Servo myservo;  // create Servo object to control a servo
// twelve Servo objects can be created on most boards
int motorPos = 0;

void setup() {
  Serial.begin(115200);
  myservo.attach(27);  // attaches the servo on pin 9 to the Servo object
  pinMode(25, OUTPUT);   // Set the pin as output
  digitalWrite(25, HIGH); // Power ON (3.3V)

}

void loop() {
  int sensorValue = analogRead(33);
  Serial.println(sensorValue);
  delay(250);
  int motorPos = map(sensorValue, 520, 3300, 0, 165);
  //520 at 180, 3300 at 165
  myservo.write(motorPos); 
  Serial.println(motorPos);
}
