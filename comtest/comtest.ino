#include <WiFi.h>
#include <WiFiClient.h>

const char* ssid = "Saar Anna";
const char* password = "mosranna";

const uint16_t port = 8090;
const char* host = "192.168.223.57";

WiFiClient client;

void setup()
{
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
      } else {
        Serial.println("Received data does not have expected number of elements");
      }
    }
  } else {
    Serial.println("Disconnected from server");
    client.stop();
    delay(1000);
    if (!client.connect(host, port)) {
      Serial.println("Reconnection to host failed");
    }
  }
  
  delay(1000);
}