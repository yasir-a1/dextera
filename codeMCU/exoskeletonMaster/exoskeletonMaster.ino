#include <esp_now.h>
#include <WiFi.h>
#include <esp_wifi.h>   // for esp_wifi_set_channel


uint8_t receiverAddress[] = {0xA0, 0xB7, 0x65, 0xDD, 0x19, 0x04};
uint8_t i = 0;


typedef struct struct_message{
  uint16_t motor1;
  uint16_t motor2;
  uint16_t motor3;
  uint16_t motor4;
  uint8_t handPosition;
} struct_message;

struct_message exoskeletonPositionData;


esp_now_peer_info_t exoskeletonPeerInfo;





void setup() {
  Serial.begin(9600);

  WiFi.mode(WIFI_STA);
    esp_wifi_set_channel(6, WIFI_SECOND_CHAN_NONE); // lock channel


  if (esp_now_init() != ESP_OK){
    Serial.print("Could not initialize ESP-NOW");
    return;
  }

  memcpy(exoskeletonPeerInfo.peer_addr, receiverAddress, 6);
  exoskeletonPeerInfo.channel = 0; 
  exoskeletonPeerInfo.encrypt = false;

  if (esp_now_add_peer(&exoskeletonPeerInfo) != ESP_OK){
    Serial.println("failed to add peer");
    return;
  }

}

void loop() {
  exoskeletonPositionData.motor1 = 50;
  exoskeletonPositionData.motor1 = 150;
  exoskeletonPositionData.motor1 = 200;
  exoskeletonPositionData.motor1 = 250;
  exoskeletonPositionData.handPosition = i;

  esp_err_t result = esp_now_send(receiverAddress, (uint8_t *) &exoskeletonPositionData, sizeof(exoskeletonPositionData));

  if (result == ESP_OK){
    Serial.println("Sent with success");
  }
  else{
    Serial.println("Error sending the data");
  }
  delay(2000);
  i++;
    

}
