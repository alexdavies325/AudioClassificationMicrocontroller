/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2019 STMicroelectronics.
  * All rights reserved.</center></h2>
  *
  * This software component is licensed by ST under BSD 3-Clause license,
  * the "License"; You may not use this file except in compliance with the
  * License. You may obtain a copy of the License at:
  *                        opensource.org/licenses/BSD-3-Clause
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "app_x-cube-ai.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include <string.h>
#include "base.h"
#include "c_speech_features.h"
#include "TestSamples.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define ADC_BUF_SIZE                   113
#define BIG_BUFFER_SIZE                64000
#define LITTLE_BUFFER_SIZE             1600
#define PRINT_BUFFER_SIZE              150
#define MAX_UINT12                     4095
#define MFCC_LENGTH                    9
#define SAMPLE_RATE                    16000
#define WINDOW_LENGTH 0.025
#define WINDOW_STEP   0.01
#define NCEPS          13
#define NFILT         26
#define NFFT          512
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
ADC_HandleTypeDef hadc1;
DMA_HandleTypeDef hdma_adc1;

CRC_HandleTypeDef hcrc;

ETH_HandleTypeDef heth;

//TIM_HandleTypeDef htim3;

UART_HandleTypeDef huart3;

PCD_HandleTypeDef hpcd_USB_OTG_FS;

/* USER CODE BEGIN PV */
uint16_t adc_buf[ADC_BUF_SIZE];
int16_t BigBuffer[BIG_BUFFER_SIZE];
int16_t LittleBuffer[LITTLE_BUFFER_SIZE];
uint8_t TakeADCReading = 0;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_ETH_Init(void);
static void MX_USART3_UART_Init(void);
//static void MX_USB_OTG_FS_PCD_Init(void);
static void MX_DMA_Init(void);
static void MX_ADC1_Init(void);
//static void MX_TIM3_Init(void);
static void MX_CRC_Init(void);
/* USER CODE BEGIN PFP */
void floatToIntRound(float *input, int16_t *output);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  /* USER CODE BEGIN 1 */
	//uint32_t startTime = 0;
	//uint32_t finishTime = 0;
	csf_float* MFCC = NULL;
	int mfcc_d1 = 0;
	int mfcc_d2 = 0;
	csf_float winFunc[400] = {0.0};
	char stringBuffer[PRINT_BUFFER_SIZE];
  /* USER CODE END 1 */
  

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_ETH_Init();
  MX_USART3_UART_Init();
  //MX_USB_OTG_FS_PCD_Init();
  MX_DMA_Init();
  MX_ADC1_Init();
  MX_CRC_Init();
  //MX_TIM3_Init();
  MX_X_CUBE_AI_Init();
  /* USER CODE BEGIN 2 */
  HAL_ADC_Start_DMA(&hadc1, (uint32_t*) adc_buf, ADC_BUF_SIZE);
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
	  // Print to serial------------------------//
	  strcpy(stringBuffer, "Recording..\r\n");
	  HAL_UART_Transmit(&huart3, (uint8_t*)stringBuffer, strlen(stringBuffer), HAL_MAX_DELAY);
	  //----------------------------------------//


	  // Gather signal, from ADC readings-----------------//
	  for (uint16_t i = 0; i < BIG_BUFFER_SIZE; i++){
		  while (TakeADCReading == 0){
			  // Do Nothing
			  // Wait to take a further ADC Reading
		  }
		  TakeADCReading = 0;
		  BigBuffer[i] = adc_buf[112];
	  }
	  //--------------------------------------------------//


	  // Print to serial---------------------------//
	  strcpy(stringBuffer, "Recorded\r\n");
	  HAL_UART_Transmit(&huart3, (uint8_t*)stringBuffer, strlen(stringBuffer), HAL_MAX_DELAY);
	  //------------------------------------------//


	  // Check Signal, from ADC, print to serial-----------------------------//
	  strcpy(stringBuffer, "Printing: \r\n");
	  HAL_UART_Transmit(&huart3, (uint8_t*)stringBuffer, strlen(stringBuffer), HAL_MAX_DELAY);

	  for (int i = 0; i < (BIG_BUFFER_SIZE/8); i++){
		  sprintf(stringBuffer, "%i, %i, %i, %i, %i, %i, %i, %i,\r\n", BigBuffer[i*8],
				  BigBuffer[i*8+1], BigBuffer[i*8+2], BigBuffer[i*8+3], BigBuffer[i*8+4],
	  			  BigBuffer[i*8+5], BigBuffer[i*8+6], BigBuffer[i*8+7]);
		  HAL_UART_Transmit(&huart3, (uint8_t*)stringBuffer, strlen(stringBuffer), HAL_MAX_DELAY);
	  }

	  strcpy(stringBuffer, "Printed \r\n");
	  HAL_UART_Transmit(&huart3, (uint8_t*)stringBuffer, strlen(stringBuffer), HAL_MAX_DELAY);
	  //---------------------------------------------------------------------//


	  // Isolate peak----------------------------//
	  // Get maximum/peak value and Index
	  int16_t maxValue = 0;
	  int16_t maxValueIndex = 0;
	  int16_t samplesBefore = (int16_t) (0.03*SAMPLE_RATE);
	  int16_t samplesAfter = (int16_t) (0.07*SAMPLE_RATE);
	  for (int i = 0; i < (BIG_BUFFER_SIZE); i++){
	  		  if (BigBuffer[i] > maxValue){
	  			  if (i < samplesBefore){
	  				  // Don't set maximum
	  			  }
	  			  else if (i > BIG_BUFFER_SIZE - samplesAfter){
	  				  // Don't set maximum
	  			  }
	  			  else {
					  maxValue = BigBuffer[i];
					  maxValueIndex = i;
	  			  }
	  		  }
	  }
	  sprintf(stringBuffer, "MaxValueIndex: %u\r\n", maxValueIndex);
	  HAL_UART_Transmit(&huart3, (uint8_t*)stringBuffer, strlen(stringBuffer), HAL_MAX_DELAY);


	  // Isolate 0.1 seconds around peak
	  // Take 0.03 seconds before peak and 0.07 seconds after
	  int16_t peakStartIndex = maxValueIndex - samplesBefore;
	  int16_t peakEndIndex = maxValueIndex + samplesAfter;

	  sprintf(stringBuffer, "peakStartIndex: %u\r\n", peakStartIndex);
	  HAL_UART_Transmit(&huart3, (uint8_t*)stringBuffer, strlen(stringBuffer), HAL_MAX_DELAY);

	  sprintf(stringBuffer, "peakEndIndex: %u\r\n", peakEndIndex);
	  HAL_UART_Transmit(&huart3, (uint8_t*)stringBuffer, strlen(stringBuffer), HAL_MAX_DELAY);


	  // Fill 0.1 second window
	  for (int16_t i = 0; (i + peakStartIndex) < peakEndIndex; i++){
		  LittleBuffer[i] = BigBuffer[i + peakStartIndex];
	  }
	  //-----------------------------------------//


	  // LittleBuffer, Before Normalization-------------------------//
	  /*strcpy(stringBuffer, "LittleBuffer, Before Normalization:\r\n");
	  HAL_UART_Transmit(&huart3, (uint8_t*)stringBuffer, strlen(stringBuffer), HAL_MAX_DELAY);

	  for (int i = 0; i < (LITTLE_BUFFER_SIZE/8); i++){
		  sprintf(stringBuffer, "%i, %i, %i, %i, %i, %i, %i, %i,\r\n", LittleBuffer[i*8],
				  LittleBuffer[i*8+1], LittleBuffer[i*8+2], LittleBuffer[i*8+3], LittleBuffer[i*8+4],
				  LittleBuffer[i*8+5], LittleBuffer[i*8+6], LittleBuffer[i*8+7]);
		  HAL_UART_Transmit(&huart3, (uint8_t*)stringBuffer, strlen(stringBuffer), HAL_MAX_DELAY);
	  }*/
	  //-----------------------------------------------------------//


	  // Calculate window function, for use in mfcc function-----------//
	  for (int i = 0; i < 400; i++){
		  winFunc[i] = 1.0;
	  }
	  //---------------------------------------------------------------//


	  // Normalization signal before mfcc--------------//
	  // Get minimum value
	  int16_t minValue = MAX_UINT12;
	  for (int i = 0; i < LITTLE_BUFFER_SIZE; i++){
		  if (LittleBuffer[i] < minValue){
			  minValue = LittleBuffer[i];
		  }
	  }

	  sprintf(stringBuffer, "MaxValue = %u\r\n", maxValue);
	  HAL_UART_Transmit(&huart3, (uint8_t*)stringBuffer, strlen(stringBuffer), HAL_MAX_DELAY);
	  sprintf(stringBuffer, "MinValue = %u\r\n", minValue);
	  HAL_UART_Transmit(&huart3, (uint8_t*)stringBuffer, strlen(stringBuffer), HAL_MAX_DELAY);

	  // Int Normalization, between 4095 and 0
	  float NormFactor = ((float) MAX_UINT12/ (float) (maxValue - minValue));
	  for (int i = 0; i < LITTLE_BUFFER_SIZE; i++){
		  float tmp = (float) (LittleBuffer[i] - minValue) * NormFactor;
		  floatToIntRound(&tmp, &LittleBuffer[i]);
	  }
	  //----------------------------------------------//


	  // Check LittleBuffer After Normalization------------------------//
	  /*strcpy(stringBuffer, "Before mfcc\r\n");
	  HAL_UART_Transmit(&huart3, (uint8_t*)stringBuffer, strlen(stringBuffer), HAL_MAX_DELAY);
	  strcpy(stringBuffer, "LittleBuffer, After Normalization:\r\n");
	  	  HAL_UART_Transmit(&huart3, (uint8_t*)stringBuffer, strlen(stringBuffer), HAL_MAX_DELAY);

	  for (int i = 0; i < (LITTLE_BUFFER_SIZE/8); i++){
		  sprintf(stringBuffer, "%i, %i, %i, %i, %i, %i, %i, %i,\r\n", LittleBuffer[i*8],
				  LittleBuffer[i*8+1], LittleBuffer[i*8+2], LittleBuffer[i*8+3], LittleBuffer[i*8+4],
				  LittleBuffer[i*8+5], LittleBuffer[i*8+6], LittleBuffer[i*8+7]);
		  HAL_UART_Transmit(&huart3, (uint8_t*)stringBuffer, strlen(stringBuffer), HAL_MAX_DELAY);
	  }*/
	  //--------------------------------------------------------------//


	  // Get Mfcc----------------------------------//
	  mfcc(LittleBuffer, LITTLE_BUFFER_SIZE, SAMPLE_RATE,
			   WINDOW_LENGTH, WINDOW_STEP, NCEPS, NFILT, NFFT,
			   0, 0, 0.97, 22, 1, winFunc,
			   &MFCC, &mfcc_d1, &mfcc_d2);
	  //-------------------------------------------//


	  // Check Mfcc, print to serial--------------------------//
	  /*strcpy(stringBuffer, "After mfcc\r\n");
	  HAL_UART_Transmit(&huart3, (uint8_t*)stringBuffer, strlen(stringBuffer), HAL_MAX_DELAY);

	  for (int i = 0; i < (MFCC_LENGTH); i++){
		  sprintf(stringBuffer, "%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\r\n", MFCC[i*13],
				  MFCC[i*13+1], MFCC[i*13+2], MFCC[i*13+3], MFCC[i*13+4], MFCC[i*13+5], MFCC[i*13+6],
				  MFCC[i*13+7], MFCC[i*13+8], MFCC[i*13+9], MFCC[i*13+10], MFCC[i*13+11],
				  MFCC[i*13+12]);
		  HAL_UART_Transmit(&huart3, (uint8_t*)stringBuffer, strlen(stringBuffer), HAL_MAX_DELAY);
	  }*/
	  //------------------------------------------------------//


	  // Initialize Input/Output buffers ----------------------//
	  ai_float NnOutput[AI_NETWORK_OUT_1_SIZE];
	  ai_float NnInput[AI_NETWORK_IN_1_SIZE];
	  //-------------------------------------------------------//


	  // Normalization after Mfcc------------------------------//
	  // Get maximum and minimum from data, for normalization
	  float maxMfccValue = 0;
	  float minMfccValue = 10000000.0;
	  for (int i = 0; i < (NCEPS*MFCC_LENGTH); i++){
		  if (MFCC[i] > maxMfccValue){
			  maxMfccValue = MFCC[i];
		  }
		  if (MFCC[i] < minMfccValue){
			  minMfccValue = MFCC[i];
		  }
	  }

	  // Normalise
	  for (int i = 0; i < (NCEPS*MFCC_LENGTH); i++){
		  NnInput[i] = (MFCC[i] - minMfccValue) / (maxMfccValue - minMfccValue);
	  }
	  //------------------------------------------------------//


	  // Check Input, print to serial--------------//
	  /*strcpy(stringBuffer, "NnInput: \r\n");
	  HAL_UART_Transmit(&huart3, (uint8_t*)stringBuffer, strlen(stringBuffer), HAL_MAX_DELAY);

	  for (int i = 0; i < (9); i++){
		  sprintf(stringBuffer, "%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\r\n", NnInput[i*13],
				  NnInput[i*13+1], NnInput[i*13+2], NnInput[i*13+3], NnInput[i*13+4], NnInput[i*13+5],
				  NnInput[i*13+6], NnInput[i*13+7], NnInput[i*13+8], NnInput[i*13+9], NnInput[i*13+10],
				  NnInput[i*13+11], NnInput[i*13+12]);
		  HAL_UART_Transmit(&huart3, (uint8_t*)stringBuffer, strlen(stringBuffer), HAL_MAX_DELAY);
	  }*/
	  //-------------------------------------------//


	  // Run NN ---------------------------//
	  aiRun(NnInput,NnOutput);
	  //aiRun(GlassSample, NnOutput); // Test
	  //-----------------------------------//


	  // Print Result Probabilites --------------------//
	  for (int i = 0; i < AI_NETWORK_OUT_1_SIZE; i++){
		  sprintf(stringBuffer, "%i Output = %f\r\n", i, NnOutput[i]);
		  HAL_UART_Transmit(&huart3, (uint8_t*)stringBuffer, strlen(stringBuffer), HAL_MAX_DELAY);
	  }
	  //-----------------------------------------------//


	  // Take only one reading------------------//
	  return 1;
	  //----------------------------------------//

	  /* USER CODE END WHILE */

  /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

void floatToIntRound(float *input, int16_t *output){
	int16_t tmp = (int16_t) *input;
	float afterDecimal = *input - (float) tmp;
	if (afterDecimal < 0.5){
		*output = tmp;
	}
	else{
		*output = tmp + 1;
	}
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
  RCC_PeriphCLKInitTypeDef PeriphClkInitStruct = {0};

  /** Configure LSE Drive Capability 
  */
  HAL_PWR_EnableBkUpAccess();
  /** Configure the main internal regulator output voltage 
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);
  /** Initializes the CPU, AHB and APB busses clocks 
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_BYPASS;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 4;
  RCC_OscInitStruct.PLL.PLLN = 216;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 9;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }
  /** Activate the Over-Drive mode 
  */
  if (HAL_PWREx_EnableOverDrive() != HAL_OK)
  {
    Error_Handler();
  }
  /** Initializes the CPU, AHB and APB busses clocks 
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_7) != HAL_OK)
  {
    Error_Handler();
  }
  PeriphClkInitStruct.PeriphClockSelection = RCC_PERIPHCLK_USART3|RCC_PERIPHCLK_CLK48;
  PeriphClkInitStruct.Usart3ClockSelection = RCC_USART3CLKSOURCE_PCLK1;
  PeriphClkInitStruct.Clk48ClockSelection = RCC_CLK48SOURCE_PLL;
  if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInitStruct) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief ADC1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_ADC1_Init(void)
{

  /* USER CODE BEGIN ADC1_Init 0 */

  /* USER CODE END ADC1_Init 0 */

  ADC_ChannelConfTypeDef sConfig = {0};

  /* USER CODE BEGIN ADC1_Init 1 */

  /* USER CODE END ADC1_Init 1 */
  /** Configure the global features of the ADC (Clock, Resolution, Data Alignment and number of conversion) 
  */
  hadc1.Instance = ADC1;
  hadc1.Init.ClockPrescaler = ADC_CLOCK_SYNC_PCLK_DIV4;
  hadc1.Init.Resolution = ADC_RESOLUTION_12B;
  hadc1.Init.ScanConvMode = ADC_SCAN_DISABLE;
  hadc1.Init.ContinuousConvMode = ENABLE;
  hadc1.Init.DiscontinuousConvMode = DISABLE;
  hadc1.Init.ExternalTrigConvEdge = ADC_EXTERNALTRIGCONVEDGE_NONE;
  hadc1.Init.ExternalTrigConv = ADC_SOFTWARE_START;
  hadc1.Init.DataAlign = ADC_DATAALIGN_RIGHT;
  hadc1.Init.NbrOfConversion = 1;
  hadc1.Init.DMAContinuousRequests = ENABLE;
  hadc1.Init.EOCSelection = ADC_EOC_SINGLE_CONV;
  if (HAL_ADC_Init(&hadc1) != HAL_OK)
  {
    Error_Handler();
  }
  /** Configure for the selected ADC regular channel its corresponding rank in the sequencer and its sample time. 
  */
  sConfig.Channel = ADC_CHANNEL_3;
  sConfig.Rank = ADC_REGULAR_RANK_1;
  sConfig.SamplingTime = ADC_SAMPLETIME_3CYCLES;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN ADC1_Init 2 */

  /* USER CODE END ADC1_Init 2 */

}
/**
  * @brief CRC Initialization Function
  * @param None
  * @retval None
  */
static void MX_CRC_Init(void)
{

  /* USER CODE BEGIN CRC_Init 0 */

  /* USER CODE END CRC_Init 0 */

  /* USER CODE BEGIN CRC_Init 1 */

  /* USER CODE END CRC_Init 1 */
  hcrc.Instance = CRC;
  hcrc.Init.DefaultPolynomialUse = DEFAULT_POLYNOMIAL_ENABLE;
  hcrc.Init.DefaultInitValueUse = DEFAULT_INIT_VALUE_ENABLE;
  hcrc.Init.InputDataInversionMode = CRC_INPUTDATA_INVERSION_NONE;
  hcrc.Init.OutputDataInversionMode = CRC_OUTPUTDATA_INVERSION_DISABLE;
  hcrc.InputDataFormat = CRC_INPUTDATA_FORMAT_BYTES;
  if (HAL_CRC_Init(&hcrc) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN CRC_Init 2 */

  /* USER CODE END CRC_Init 2 */

}

/**
  * @brief ETH Initialization Function
  * @param None
  * @retval None
  */
static void MX_ETH_Init(void)
{

  /* USER CODE BEGIN ETH_Init 0 */

  /* USER CODE END ETH_Init 0 */

  /* USER CODE BEGIN ETH_Init 1 */

  /* USER CODE END ETH_Init 1 */
  heth.Instance = ETH;
  heth.Init.AutoNegotiation = ETH_AUTONEGOTIATION_ENABLE;
  heth.Init.PhyAddress = LAN8742A_PHY_ADDRESS;
  heth.Init.MACAddr[0] =   0x00;
  heth.Init.MACAddr[1] =   0x80;
  heth.Init.MACAddr[2] =   0xE1;
  heth.Init.MACAddr[3] =   0x00;
  heth.Init.MACAddr[4] =   0x00;
  heth.Init.MACAddr[5] =   0x00;
  heth.Init.RxMode = ETH_RXPOLLING_MODE;
  heth.Init.ChecksumMode = ETH_CHECKSUM_BY_HARDWARE;
  heth.Init.MediaInterface = ETH_MEDIA_INTERFACE_RMII;

  /* USER CODE BEGIN MACADDRESS */
    
  /* USER CODE END MACADDRESS */

  if (HAL_ETH_Init(&heth) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN ETH_Init 2 */

  /* USER CODE END ETH_Init 2 */

}

/**
  * @brief USART3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART3_UART_Init(void)
{

  /* USER CODE BEGIN USART3_Init 0 */

  /* USER CODE END USART3_Init 0 */

  /* USER CODE BEGIN USART3_Init 1 */

  /* USER CODE END USART3_Init 1 */
  huart3.Instance = USART3;
  huart3.Init.BaudRate = 115200;
  huart3.Init.WordLength = UART_WORDLENGTH_8B;
  huart3.Init.StopBits = UART_STOPBITS_1;
  huart3.Init.Parity = UART_PARITY_NONE;
  huart3.Init.Mode = UART_MODE_TX_RX;
  huart3.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart3.Init.OverSampling = UART_OVERSAMPLING_16;
  huart3.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart3.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart3) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART3_Init 2 */

  /* USER CODE END USART3_Init 2 */

}

/**
  * @brief USB_OTG_FS Initialization Function
  * @param None
  * @retval None
  */
/*static void MX_USB_OTG_FS_PCD_Init(void)
{

   USER CODE BEGIN USB_OTG_FS_Init 0

   USER CODE END USB_OTG_FS_Init 0

   USER CODE BEGIN USB_OTG_FS_Init 1

   USER CODE END USB_OTG_FS_Init 1
  hpcd_USB_OTG_FS.Instance = USB_OTG_FS;
  hpcd_USB_OTG_FS.Init.dev_endpoints = 6;
  hpcd_USB_OTG_FS.Init.speed = PCD_SPEED_FULL;
  hpcd_USB_OTG_FS.Init.dma_enable = DISABLE;
  hpcd_USB_OTG_FS.Init.phy_itface = PCD_PHY_EMBEDDED;
  hpcd_USB_OTG_FS.Init.Sof_enable = ENABLE;
  hpcd_USB_OTG_FS.Init.low_power_enable = DISABLE;
  hpcd_USB_OTG_FS.Init.lpm_enable = DISABLE;
  hpcd_USB_OTG_FS.Init.vbus_sensing_enable = ENABLE;
  hpcd_USB_OTG_FS.Init.use_dedicated_ep1 = DISABLE;
  if (HAL_PCD_Init(&hpcd_USB_OTG_FS) != HAL_OK)
  {
    Error_Handler();
  }
   USER CODE BEGIN USB_OTG_FS_Init 2

   USER CODE END USB_OTG_FS_Init 2

}*/

/** 
  * Enable DMA controller clock
  */
static void MX_DMA_Init(void) 
{

  /* DMA controller clock enable */
  __HAL_RCC_DMA2_CLK_ENABLE();

  /* DMA interrupt init */
  /* DMA2_Stream0_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA2_Stream0_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA2_Stream0_IRQn);

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();
  __HAL_RCC_GPIOG_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOB, LD1_Pin|LD3_Pin|LD2_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(USB_PowerSwitchOn_GPIO_Port, USB_PowerSwitchOn_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin : USER_Btn_Pin */
  GPIO_InitStruct.Pin = USER_Btn_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_RISING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(USER_Btn_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : LD1_Pin LD3_Pin LD2_Pin */
  GPIO_InitStruct.Pin = LD1_Pin|LD3_Pin|LD2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /*Configure GPIO pin : USB_PowerSwitchOn_Pin */
  GPIO_InitStruct.Pin = USB_PowerSwitchOn_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(USB_PowerSwitchOn_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : USB_OverCurrent_Pin */
  GPIO_InitStruct.Pin = USB_OverCurrent_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(USB_OverCurrent_GPIO_Port, &GPIO_InitStruct);

}

/* USER CODE BEGIN 4 */
// Called when buffer is completely filled
void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef* hadc){
	TakeADCReading = 1;
}
/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */

  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{ 
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     tex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
