#include <esp_now.h>
#include <WiFi.h>
#include <esp_wifi.h>   // for esp_wifi_set_channel


uint8_t receiverAddress[] = {0xA0, 0xB7, 0x65, 0xDD, 0x19, 0x04};
uint8_t i = 0;

const uint8_t pot1 = 32;
const uint8_t pot2 = 33;
const uint8_t pot3 = 34;
const uint8_t pot4 = 35;

float pot1Value = 0;
float pot2Value = 0;
float pot3Value = 0;
float pot4Value = 0;


uint8_t currentHandPosition = 0;
String InBytes;


typedef struct struct_message{
  uint16_t motor1;
  uint16_t motor2;
  uint16_t motor3;
  uint16_t motor4;
  uint8_t handPosition;
} struct_message;

struct_message exoskeletonPositionData;


esp_now_peer_info_t exoskeletonPeerInfo;



void readPotValues(){
  pot1Value = analogRead(pot1);
  pot2Value = analogRead(pot2);
  pot3Value = analogRead(pot3);
  pot4Value = analogRead(pot4);

}


//Change sampling to double of human hand, set Int on the line???
void handPositionRead(){
  if (Serial.available() > 0){
    InBytes = Serial.readStringUntil('\n');
    if (InBytes == "OPEN"){
      currentHandPosition = 1;
    }
    else if (InBytes == "PINCH"){
      currentHandPosition = 2;
    }
    else if (InBytes == "FIST"){
      currentHandPosition = 3;
    }
    else{
      currentHandPosition = 0;
    }
}

void buildMessage(){
  exoskeletonPositionData.motor1 = map(pot1Value, 0, 4095, 0, 180);
  exoskeletonPositionData.motor2 = map(pot2Value, 0, 4095, 0, 180);
  exoskeletonPositionData.motor3 = map(pot3Value, 0, 4095, 0, 180);
  exoskeletonPositionData.motor4 = map(pot4Value, 0, 4095, 0, 180);
  exoskeletonPositionData.handPosition = currentHandPosition;
}

void setup() {
  Serial.begin(9600);
  WiFi.mode(WIFI_STA);
  esp_wifi_set_channel(6, WIFI_SECOND_CHAN_NONE); // lock channel
  analogReadResolution(12);


  if (esp_now_init() != ESP_OK){
    Serial.print("Could not initialize ESP-NOW");
    return;
  }
  
  memset(&exoskeletonPeerInfo, 0, sizeof(exoskeletonPeerInfo));
  memcpy(exoskeletonPeerInfo.peer_addr, receiverAddress, 6);
  exoskeletonPeerInfo.channel = 0; 
  exoskeletonPeerInfo.encrypt = false;

  if (esp_now_add_peer(&exoskeletonPeerInfo) != ESP_OK){
    Serial.println("failed to add peer");
    return;
  }



}

void loop() {

  readPotValues();
  if (Serial.available() > 0){  //DO I run the risk of missing messages, or will messages remain on bus til read?
    handPositionRead();
  }

  buildMessage();

  esp_err_t result = esp_now_send(receiverAddress, (uint8_t *) &exoskeletonPositionData, sizeof(exoskeletonPositionData));

  if (result == ESP_OK){
    Serial.println("Sent with success");
  }
  else{
    Serial.println("Error sending the data");
  }
  delay(50);
    

}
