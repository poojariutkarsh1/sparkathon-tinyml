#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

Adafruit_MPU6050 mpu;

#define IR_PIN 27

void setup() {
  Serial.begin(115200);

  delay(1000);

  Serial.println();
  Serial.println("================================");
  Serial.println("WAREHOUSE SENSOR TEST");
  Serial.println("================================");

  // -----------------------------
  // IR SENSOR
  // -----------------------------
  pinMode(IR_PIN, INPUT);

  Serial.println("IR sensor initialized");

  // -----------------------------
  // I2C
  // -----------------------------
  Wire.begin(21, 22);

  // -----------------------------
  // MPU6050
  // -----------------------------
  Serial.println("Checking MPU6050...");

  if (!mpu.begin()) {
    Serial.println("ERROR: MPU6050 NOT detected!");
    Serial.println("Check VCC, GND, SDA and SCL.");
    
    while (1) {
      delay(1000);
    }
  }

  Serial.println("MPU6050 detected!");

  // Set accelerometer range
  mpu.setAccelerometerRange(MPU6050_RANGE_2_G);

  // Set gyroscope range
  mpu.setGyroRange(MPU6050_RANGE_250_DEG);

  // Set filter bandwidth
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  Serial.println("IR sensor detected");
  Serial.println();
  Serial.println("Starting sensor readings...");
  Serial.println("--------------------------------");
}

void loop() {

  sensors_event_t a, g, temp;

  // Read MPU6050
  mpu.getEvent(&a, &g, &temp);

  // Read IR
  int irState = digitalRead(IR_PIN);

  // -----------------------------
  // Accelerometer
  // -----------------------------
  Serial.println("Accelerometer:");

  Serial.print("X = ");
  Serial.print(a.acceleration.x);
  Serial.println(" m/s^2");

  Serial.print("Y = ");
  Serial.print(a.acceleration.y);
  Serial.println(" m/s^2");

  Serial.print("Z = ");
  Serial.print(a.acceleration.z);
  Serial.println(" m/s^2");

  // -----------------------------
  // Gyroscope
  // -----------------------------
  Serial.println("Gyroscope:");

  Serial.print("X = ");
  Serial.print(g.gyro.x);
  Serial.println(" rad/s");

  Serial.print("Y = ");
  Serial.print(g.gyro.y);
  Serial.println(" rad/s");

  Serial.print("Z = ");
  Serial.print(g.gyro.z);
  Serial.println(" rad/s");

  // -----------------------------
  // IR
  // -----------------------------
  Serial.print("IR: ");

  if (irState == HIGH) {
    Serial.println("DETECTED");
  } else {
    Serial.println("CLEAR");
  }

  Serial.println("--------------------------------");

  delay(500);
}
