//comms: client
#include <WiFi.h>
#include <WiFiClient.h>

const char* ssid = "mamba";
const char* password = "happy100";

const uint16_t port = 8090;
const char* host = "192.168.43.62";

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
      String elements[10]; // Assuming maximum of 10 elements
      for (int i = 0; i < receivedData.length(); i++) {
        if (receivedData.charAt(i) == ',') {
          numElements++;
        } else {
          elements[numElements] += receivedData.charAt(i);
        }
      }
      numElements++;
      
      // Print elements separately
      Serial.println("Elements:");
      for (int i = 0; i < numElements; i++) {
        Serial.print("Item ");
        Serial.print(i);
        Serial.print(": ");
        Serial.println(elements[i]);
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
