String InBytes;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  pinMode(13, OUTPUT);
  
}

void blink (const String& blinkState){
  if (blinkState == "FIST"){
    digitalWrite(13, HIGH);
    delay (50);
    digitalWrite(13, LOW);
    delay (50);
    digitalWrite(13, HIGH);
    delay (50);
    digitalWrite(13, LOW);
    delay (50);
    digitalWrite(13, HIGH);
    delay (50);
    digitalWrite(13, LOW);
    delay (50);
  }
  else if (blinkState == "OPEN"){
    digitalWrite(13, HIGH);
    delay (250);
    digitalWrite(13, LOW);
    delay (250);
    digitalWrite(13, HIGH);
    delay (250);
    digitalWrite(13, LOW);
    delay (250);
    digitalWrite(13, HIGH);
    delay (250);
    digitalWrite(13, LOW);
    delay (250);
  }
  else if (blinkState == "PINCH"){
    digitalWrite(13, HIGH);
    delay (500);
    digitalWrite(13, LOW);
    delay (500);
    digitalWrite(13, HIGH);
    delay (500);
    digitalWrite(13, LOW);
    delay (500);
    digitalWrite(13, HIGH);
    delay (500);
    digitalWrite(13, LOW);
    delay (500);
  }
  else{
    digitalWrite(13, LOW);
  }


}

void loop() {
  if (Serial.available() > 0){
    InBytes = Serial.readStringUntil('\n');
    blink(InBytes);
  }

}
