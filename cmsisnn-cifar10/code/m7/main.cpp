#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "arm_math.h"
#include "parameter.h"
#include "weights.h"
#include "arm_nnfunctions.h"

static q7_t conv1_wt[CONV1_IN_CH*CONV1_KER_DIM*CONV1_KER_DIM*CONV1_OUT_CH] = CONV1_WT;
static q7_t conv1_bias[CONV1_OUT_CH] = CONV1_BIAS;

static q7_t conv2_wt[CONV2_IN_CH*CONV2_KER_DIM*CONV2_KER_DIM*CONV2_OUT_CH] = CONV2_WT;
static q7_t conv2_bias[CONV2_OUT_CH] = CONV2_BIAS;

static q7_t conv3_wt[CONV3_IN_CH*CONV3_KER_DIM*CONV3_KER_DIM*CONV3_OUT_CH] = CONV3_WT;
static q7_t conv3_bias[CONV3_OUT_CH] = CONV3_BIAS;

static q7_t ip1_wt[IP1_IN_DIM*IP1_OUT_DIM] = IP1_WT;
static q7_t ip1_bias[IP1_OUT_DIM] = IP1_BIAS;

q7_t input_data[DATA_OUT_CH*DATA_OUT_DIM*DATA_OUT_DIM];
q7_t output_data[IP1_OUT_DIM];

q7_t col_buffer[6400];
q7_t scratch_buffer[40960];

void save(const char* name, q7_t* data, size_t size)
{
    FILE* fp = fopen(name, "wb");
    assert(fp);
    fwrite(data, size, 1, fp);
    fclose(fp);
}

void run_nn() {

  q7_t* buffer1 = scratch_buffer;
  q7_t* buffer2 = buffer1 + 32768;
  save("out/data.raw", input_data,DATA_OUT_CH*DATA_OUT_DIM*DATA_OUT_DIM);
  arm_convolve_HWC_q7_RGB(input_data, CONV1_IN_DIM, CONV1_IN_CH, conv1_wt, CONV1_OUT_CH, CONV1_KER_DIM, CONV1_PAD, CONV1_STRIDE, conv1_bias, CONV1_BIAS_LSHIFT, CONV1_OUT_RSHIFT, buffer1, CONV1_OUT_DIM, (q15_t*)col_buffer, NULL);
  save("out/conv1.raw", buffer1,CONV1_OUT_CH*CONV1_OUT_DIM*CONV1_OUT_DIM);
  arm_maxpool_q7_HWC(buffer1, POOL1_IN_DIM, POOL1_IN_CH, POOL1_KER_DIM, POOL1_PAD, POOL1_STRIDE, POOL1_OUT_DIM, col_buffer, buffer2);
  save("out/pool1.raw", buffer2,CONV1_OUT_CH*POOL1_OUT_DIM*POOL1_OUT_DIM);
  arm_relu_q7(buffer2, RELU1_OUT_DIM*RELU1_OUT_DIM*RELU1_OUT_CH);
  arm_convolve_HWC_q7_fast(buffer2, CONV2_IN_DIM, CONV2_IN_CH, conv2_wt, CONV2_OUT_CH, CONV2_KER_DIM, CONV2_PAD, CONV2_STRIDE, conv2_bias, CONV2_BIAS_LSHIFT, CONV2_OUT_RSHIFT, buffer1, CONV2_OUT_DIM, (q15_t*)col_buffer, NULL);
  save("out/conv2.raw", buffer1,CONV2_OUT_CH*CONV2_OUT_DIM*CONV2_OUT_DIM);
  arm_relu_q7(buffer1, RELU2_OUT_DIM*RELU2_OUT_DIM*RELU2_OUT_CH);
  arm_avepool_q7_HWC(buffer1, POOL2_IN_DIM, POOL2_IN_CH, POOL2_KER_DIM, POOL2_PAD, POOL2_STRIDE, POOL2_OUT_DIM, col_buffer, buffer2);
  save("out/pool2.raw", buffer2,CONV2_OUT_CH*POOL2_OUT_DIM*POOL2_OUT_DIM);
  arm_convolve_HWC_q7_fast(buffer2, CONV3_IN_DIM, CONV3_IN_CH, conv3_wt, CONV3_OUT_CH, CONV3_KER_DIM, CONV3_PAD, CONV3_STRIDE, conv3_bias, CONV3_BIAS_LSHIFT, CONV3_OUT_RSHIFT, buffer1, CONV3_OUT_DIM, (q15_t*)col_buffer, NULL);
  save("out/conv3.raw", buffer1,CONV3_OUT_CH*CONV3_OUT_DIM*CONV3_OUT_DIM);
  arm_relu_q7(buffer1, RELU3_OUT_DIM*RELU3_OUT_DIM*RELU3_OUT_CH);
  arm_avepool_q7_HWC(buffer1, POOL3_IN_DIM, POOL3_IN_CH, POOL3_KER_DIM, POOL3_PAD, POOL3_STRIDE, POOL3_OUT_DIM, col_buffer, buffer2);
  save("out/pool3.raw", buffer2,CONV3_OUT_CH*POOL3_OUT_DIM*POOL3_OUT_DIM);
  arm_fully_connected_q7_opt(buffer2, ip1_wt, IP1_IN_DIM, IP1_OUT_DIM, IP1_BIAS_LSHIFT, IP1_OUT_RSHIFT, ip1_bias, output_data, (q15_t*)col_buffer);
  save("out/ip1.raw", output_data,IP1_OUT_DIM);
  arm_softmax_q7(output_data, 10, output_data);
  save("out/prob.raw", output_data,IP1_OUT_DIM);
}

int main (int argc, char* argv[]) {
  if(argc > 1) {
    FILE* fp = fopen(argv[1],"rb");
    if(NULL != fp) {
      fread(input_data, 1, sizeof(input_data), fp);
      fclose(fp);
    } else {
      printf("file %s not exists.\n", argv[1]);
      return -2;
    }
  } else {
    printf("usage: %s img.bin\n", argv[0]);
    return -1;
  }

  /* input pre-processing */
  #define INPUT_MEAN_SHIFT {125,123,114}
  #define INPUT_RIGHT_SHIFT {8,8,8}
  int mean_data[3] = INPUT_MEAN_SHIFT;
  unsigned int scale_data[3] = INPUT_RIGHT_SHIFT;
  for (int i=0;i<32*32*3; i+=3) {
    input_data[i] =   (q7_t)__SSAT( ((((int)input_data[i]   - mean_data[0])<<7) + (0x1<<(scale_data[0]-1)))
                             >> scale_data[0], 8);
    input_data[i+1] = (q7_t)__SSAT( ((((int)input_data[i+1] - mean_data[1])<<7) + (0x1<<(scale_data[1]-1)))
                             >> scale_data[1], 8);
    input_data[i+2] = (q7_t)__SSAT( ((((int)input_data[i+2] - mean_data[2])<<7) + (0x1<<(scale_data[2]-1)))
                             >> scale_data[2], 8);
  }

  run_nn();

  static const char* CIFAR10_LABELS_LIST[] = { "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck" };
  for (int i = 0; i < 10; i++)
  {
      /* Q to Float: Q7*2^-7 */
      printf("%s: %d.%d%%\n", CIFAR10_LABELS_LIST[i], output_data[i]*100/128, (output_data[i]*1000/128)%10);
  }

  return 0;
}
