#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

#include <Chirale_TensorFlowLite.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "warehouse_model.h"

// ======================================
// MPU6050
// ======================================

#define SDA_PIN 21
#define SCL_PIN 22

Adafruit_MPU6050 mpu;

// ======================================
// TinyML
// ======================================

constexpr int NUM_SAMPLES = 100;
constexpr int NUM_FEATURES = 6;

constexpr int kTensorArenaSize = 20 * 1024;

alignas(16) uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// ======================================
// Training normalization
// ======================================

const float meanValues[NUM_FEATURES] = {
  -0.969311,
   0.414217,
   9.752275,
   0.002164,
   0.025100,
  -0.008800
};

const float stdValues[NUM_FEATURES] = {
  2.352505,
  0.552999,
  0.936201,
  0.297388,
  1.357119,
  0.290040
};

// ======================================
// Model quantization
// ======================================

const float INPUT_SCALE = 0.042450081557035446;
const int INPUT_ZERO_POINT = -4;

const float OUTPUT_SCALE = 0.00390625;
const int OUTPUT_ZERO_POINT = -128;

// ======================================
// Classes
// ======================================

const char* classNames[3] = {
  "NORMAL",
  "STATIONARY",
  "ABNORMAL_VIBRATION"
};

// ======================================
// Setup
// ======================================

void setup() {

  Serial.begin(115200);
  delay(2000);

  Serial.println();
  Serial.println("======================================");
  Serial.println(" WAREHOUSE TinyML ESP32");
  Serial.println("======================================");

  // ====================================
  // Initialize MPU6050
  // ====================================

  Serial.println();
  Serial.println("Initializing MPU6050...");

  Wire.begin(SDA_PIN, SCL_PIN);

  if (!mpu.begin()) {

    Serial.println("ERROR: MPU6050 NOT DETECTED!");

    while (1) {
      delay(1000);
    }
  }

  Serial.println("MPU6050 detected.");

  mpu.setAccelerometerRange(MPU6050_RANGE_2_G);
  mpu.setGyroRange(MPU6050_RANGE_250_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  // ====================================
  // Load TinyML model
  // ====================================

  Serial.println();
  Serial.println("Loading TinyML model...");

  model = tflite::GetModel(warehouse_cnn_int8_model);

  if (model->version() != TFLITE_SCHEMA_VERSION) {

    Serial.println("ERROR: MODEL SCHEMA VERSION MISMATCH!");

    while (1) {
      delay(1000);
    }
  }

  // ====================================
  // Create interpreter
  // ====================================

  static tflite::AllOpsResolver resolver;

  static tflite::MicroInterpreter static_interpreter(
    model,
    resolver,
    tensor_arena,
    kTensorArenaSize
  );

  interpreter = &static_interpreter;

  // ====================================
  // Allocate tensors
  // ====================================

  Serial.println("Allocating tensors...");

  if (interpreter->AllocateTensors() != kTfLiteOk) {

    Serial.println("ERROR: AllocateTensors() FAILED!");

    while (1) {
      delay(1000);
    }
  }

  // ====================================
  // Get input/output tensors
  // ====================================

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("TinyML model loaded successfully.");

  Serial.println();
  Serial.print("Input type: ");
  Serial.println(input->type);

  Serial.print("Input bytes: ");
  Serial.println(input->bytes);

  Serial.print("Output type: ");
  Serial.println(output->type);

  Serial.print("Output bytes: ");
  Serial.println(output->bytes);

  Serial.println();
  Serial.println("======================================");
  Serial.println(" READY FOR INFERENCE");
  Serial.println("======================================");

  delay(2000);
}

// ======================================
// Main loop
// ======================================

void loop() {

  Serial.println();
  Serial.println("--------------------------------------");
  Serial.println("Collecting 2-second sensor window...");
  Serial.println("--------------------------------------");

  // ====================================
  // Collect 100 samples
  // ====================================

  for (int i = 0; i < NUM_SAMPLES; i++) {

    sensors_event_t accel;
    sensors_event_t gyro;
    sensors_event_t temp;

    mpu.getEvent(&accel, &gyro, &temp);

    float rawValues[NUM_FEATURES] = {

      accel.acceleration.x,
      accel.acceleration.y,
      accel.acceleration.z,

      gyro.gyro.x,
      gyro.gyro.y,
      gyro.gyro.z
    };

    // ==================================
    // Normalize + INT8 quantization
    // ==================================

    for (int feature = 0; feature < NUM_FEATURES; feature++) {

      float normalized =
        (rawValues[feature] - meanValues[feature])
        / stdValues[feature];

      int quantized =
        round(normalized / INPUT_SCALE)
        + INPUT_ZERO_POINT;

      // Clamp to INT8 range
      if (quantized > 127) {
        quantized = 127;
      }

      if (quantized < -128) {
        quantized = -128;
      }

      int index =
        i * NUM_FEATURES + feature;

      input->data.int8[index] =
        (int8_t)quantized;
    }

    delay(20);
  }

  Serial.println("Window collected.");

  // ====================================
  // Run inference
  // ====================================

  Serial.println("Running inference...");

  unsigned long startTime = micros();

  TfLiteStatus status =
    interpreter->Invoke();

  unsigned long inferenceTime =
    micros() - startTime;

  if (status != kTfLiteOk) {

    Serial.println("ERROR: INFERENCE FAILED!");

    delay(2000);
    return;
  }

  Serial.println("Inference complete.");

  // ====================================
  // Read outputs
  // ====================================

  float probabilities[3];

  for (int i = 0; i < 3; i++) {

    int8_t rawOutput =
      output->data.int8[i];

    probabilities[i] =
      (rawOutput - OUTPUT_ZERO_POINT)
      * OUTPUT_SCALE;

    // Keep result inside normal probability range
    if (probabilities[i] < 0.0) {
      probabilities[i] = 0.0;
    }

    if (probabilities[i] > 1.0) {
      probabilities[i] = 1.0;
    }
  }

  // ====================================
  // Find highest probability
  // ====================================

  int predictedClass = 0;

  for (int i = 1; i < 3; i++) {

    if (probabilities[i] >
        probabilities[predictedClass]) {

      predictedClass = i;
    }
  }

  // ====================================
  // Print results
  // ====================================

  Serial.println();
  Serial.println("======================================");
  Serial.println("         TINYML RESULT");
  Serial.println("======================================");

  Serial.print("NORMAL:              ");
  Serial.print(probabilities[0] * 100.0, 2);
  Serial.println("%");

  Serial.print("STATIONARY:          ");
  Serial.print(probabilities[1] * 100.0, 2);
  Serial.println("%");

  Serial.print("ABNORMAL_VIBRATION:  ");
  Serial.print(probabilities[2] * 100.0, 2);
  Serial.println("%");

  Serial.println();

  Serial.print("PREDICTED CLASS: ");
  Serial.println(classNames[predictedClass]);

  Serial.print("INFERENCE TIME: ");
  Serial.print(inferenceTime / 1000.0, 2);
  Serial.println(" ms");

  Serial.println("======================================");

  delay(2000);
}