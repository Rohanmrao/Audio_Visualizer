#include <WiFi.h>
#include <WiFiClient.h>
#include <Adafruit_NeoPixel.h>

const char* ssid = "";
const char* password = "";

const uint16_t port = ;
const char* host = ""; // Server laptop

#define LED_PIN 4
#define LED_COUNT 60

WiFiClient client;

uint8_t classic_leds[60][3] = {
    {60, 0, 80}, {64, 0, 88}, {68, 0, 92}, {76, 0, 100}, {80, 0, 108}, {84, 0, 116},
    {92, 0, 120}, {96, 0, 128}, {100, 0, 136}, {104, 0, 140}, {112, 0, 148}, {116, 0, 156},
    {124, 8, 164}, {132, 24, 168}, {140, 40, 172}, {148, 56, 180}, {156, 72, 184},
    {164, 88, 192}, {172, 104, 196}, {180, 120, 204}, {192, 136, 208}, {200, 152, 216},
    {208, 168, 220}, {216, 184, 224}, {224, 200, 232}, {232, 216, 236}, {240, 232, 244},
    {248, 248, 248}, {252, 252, 244}, {252, 252, 228}, {252, 252, 212}, {252, 252, 196},
    {252, 252, 180}, {252, 252, 164}, {252, 252, 148}, {252, 252, 132}, {252, 252, 116},
    {252, 252, 100}, {252, 252, 84}, {252, 252, 68}, {252, 252, 52}, {252, 252, 36},
    {252, 252, 20}, {252, 252, 4}, {244, 240, 0}, {232, 224, 8}, {220, 208, 12},
    {208, 192, 16}, {196, 176, 24}, {184, 160, 28}, {172, 144, 32}, {160, 128, 36},
    {144, 112, 44}, {132, 96, 48}, {120, 80, 52}, {108, 64, 60}, {96, 48, 64},
    {84, 32, 68}, {72, 16, 76}, {60, 0, 80}
  };

uint8_t jazz_leds[60][3] ={{244, 184, 0}, {240, 180, 0},
      {236, 176, 0}, {232, 172, 0}, {228, 168, 0}, {224, 164, 0},
      {220, 160, 0}, {216, 156, 0}, {212, 152, 0}, {208, 148, 0},
      {204, 144, 0}, {200, 140, 0}, {196, 136, 0}, {192, 132, 0},
      {188, 128, 0}, {184, 124, 0}, {180, 120, 0}, {176, 116, 0},
      {172, 112, 0}, {168, 108, 0}, {164, 104, 0}, {160, 100, 0},
      {156, 96, 0}, {152, 92, 0}, {148, 88, 0}, {144, 84, 0},
      {140, 80, 0}, {136, 76, 0}, {132, 72, 0}, {128, 68, 0},
      {124, 64, 0}, {120, 60, 0}, {116, 56, 0}, {112, 52, 0},
      {108, 48, 0}, {104, 44, 0}, {100, 40, 0}, {96, 36, 0},
      {92, 32, 0}, {88, 28, 0}, {84, 24, 0}, {80, 20, 0},
      {76, 16, 0}, {72, 12, 0}, {68, 8, 0}, {64, 4, 0},
      {60, 0, 0}, {56, 0, 0}, {52, 0, 0}, {48, 0, 0},
      {44, 0, 0}, {40, 0, 0}, {36, 0, 0}, {32, 0, 0},
      {28, 0, 0}, {24, 0, 0}, {20, 0, 0}, {16, 0, 0},
      {12, 0, 0}, {8, 0, 0}};

uint8_t metal_leds[60][3] ={{244, 184, 0}, {240, 180, 0},
      {236, 176, 0}, {232, 172, 0}, {228, 168, 0}, {224, 164, 0},
      {220, 160, 0}, {216, 156, 0}, {212, 152, 0}, {208, 148, 0},
      {204, 144, 0}, {200, 140, 0}, {196, 136, 0}, {192, 132, 0},
      {188, 128, 0}, {184, 124, 0}, {180, 120, 0}, {176, 116, 0},
      {172, 112, 0}, {168, 108, 0}, {164, 104, 0}, {160, 100, 0},
      {156, 96, 0}, {152, 92, 0}, {148, 88, 0}, {144, 84, 0},
      {140, 80, 0}, {136, 76, 0}, {132, 72, 0}, {128, 68, 0},
      {124, 64, 0}, {120, 60, 0}, {116, 56, 0}, {112, 52, 0},
      {108, 48, 0}, {104, 44, 0}, {100, 40, 0}, {96, 36, 0},
      {92, 32, 0}, {88, 28, 0}, {84, 24, 0}, {80, 20, 0},
      {76, 16, 0}, {72, 12, 0}, {68, 8, 0}, {64, 4, 0},
      {60, 0, 0}, {56, 0, 0}, {52, 0, 0}, {48, 0, 0},
      {44, 0, 0}, {40, 0, 0}, {36, 0, 0}, {32, 0, 0},
      {28, 0, 0}, {24, 0, 0}, {20, 0, 0}, {16, 0, 0},
      {12, 0, 0}, {8, 0, 0}};

Adafruit_NeoPixel pixels(LED_COUNT, LED_PIN, NEO_GRB + NEO_KHZ800);

void classic_beat(float tempoval){

    float tempo = tempoval;
    for (int i = 0; i < LED_COUNT; i++) {
      pixels.setPixelColor(i, pixels.Color(classic_leds[i][0], classic_leds[i][1], classic_leds[i][2]));
      pixels.show();
      delay(tempo/3);
    }

    for (int i = LED_COUNT - 1; i >= 0; i--) {
      pixels.setPixelColor(i, pixels.Color(0, 0, 0));
      pixels.show();
      delay(tempo/3);
    }

  }

void jazz_beat(float tempoval){

    float tempo = tempoval;
    for (int i = 0; i < LED_COUNT; i++) {
      pixels.setPixelColor(i, pixels.Color(jazz_leds[i][0], jazz_leds[i][1], jazz_leds[i][2]));
      pixels.show();
      delay(tempo/3);
    }
    for (int i = LED_COUNT - 1; i >= 0; i--) {
      pixels.setPixelColor(i, pixels.Color(0, 0, 0));
      pixels.show();
      delay(tempo/3);

    }

}

void metal_beat(float tempoval){
    float tempo = tempoval;

    for (int i = 0; i < LED_COUNT; i++) {
      pixels.setPixelColor(i, pixels.Color(metal_leds[i][0], metal_leds[i][1], metal_leds[i][2]));
      pixels.show();
      delay(tempo/3);
    }
    for (int i = LED_COUNT - 1; i >= 0; i--) {
      pixels.setPixelColor(i, pixels.Color(0, 0, 0));
      pixels.show();
      delay(tempo/3);
    }
  }


void setup()
{
  pixels.begin();
  pixels.setBrightness(30); // set the brightness of the LED's
  Serial.begin(115200);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.println("...");
  }

  Serial.print("WiFi connected with IP: ");
  Serial.println(WiFi.localIP());

  Serial.print("Connecting to host: ");
  Serial.println(host);
  
  if (!client.connect(host, port)) {
    Serial.println("Connection to host failed");
  }
}

void loop()
{
  if (client.connected()) {
    String receivedData;
    while (client.available()) {
      char c = client.read();
      receivedData += c;
    }
    
    if (!receivedData.isEmpty()) {
      Serial.print("Received data: ");
      Serial.println(receivedData);
      
      // Split received data by comma
      int numElements = 0;
      String elements[2]; // Assuming 2 elements: string and float
      String currentElement;
      for (int i = 0; i < receivedData.length(); i++) {
        if (receivedData.charAt(i) == ',') {
          elements[numElements] = currentElement;
          currentElement = "";
          numElements++;
        } else {
          currentElement += receivedData.charAt(i);
        }
      }
      // Process the last element after the loop
      if (!currentElement.isEmpty()) {
        elements[numElements] = currentElement;
        numElements++;
      }
      
      // Check if the expected number of elements is received
      if (numElements == 2) {
        String strElement = elements[0];
        float floatElement = elements[1].toFloat(); // Convert string to float
        Serial.println("Elements:");
        Serial.print("String: ");
        Serial.println(strElement);
        Serial.print("Float: ");
        Serial.println(floatElement);

        
        if (strElement == "Jazz"){jazz_beat(floatElement);}

        if (strElement == "Metal"){metal_beat(floatElement);}

        if (strElement == "Classical"){classic_beat(floatElement);}
      } 
      
      else {Serial.println("Received data does not have expected number of elements");}
    }
  } 
  
  else 
  {
      Serial.println("Disconnected from server");
      client.stop();
      delay(1000);
      
      if (!client.connect(host, port)) {Serial.println("Reconnection to host failed");}
    }
    
  delay(1000);

}
