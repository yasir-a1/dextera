#include <esp_now.h>
#include <WiFi.h>
#include <esp_wifi.h>   // for esp_wifi_set_channel


typedef struct struct_message{
  uint16_t motor1;
  uint16_t motor2;
  uint16_t motor3;
  uint16_t motor4;
  uint8_t handPosition;
} struct_message;

struct_message robotPositionData;

void dataRecvCallback(const esp_now_recv_info* info, const uint8_t *incomingData, int len){
  memcpy(&robotPositionData, incomingData, sizeof(robotPositionData));
  Serial.print("Bytes received: ");
  Serial.println(len);
  Serial.print("motor1: ");
  Serial.println(robotPositionData.motor1);
  Serial.print("motor2: ");
  Serial.println(robotPositionData.motor2);
  Serial.print("motor3: ");
  Serial.println(robotPositionData.motor3);
  Serial.print("motor4: ");
  Serial.println(robotPositionData.motor4);
  Serial.print("handPosition: ");
  Serial.println(robotPositionData.handPosition);
  Serial.println();
}


void setup() {
  Serial.begin(9600);
  WiFi.mode(WIFI_STA);
  esp_wifi_set_channel(6, WIFI_SECOND_CHAN_NONE); // lock channel

  if (esp_now_init() != ESP_OK){
    Serial.println("Error initalizing ESP-NOW");
    return;
  }

  esp_now_register_recv_cb(dataRecvCallback);
}

void loop() {
  

}
